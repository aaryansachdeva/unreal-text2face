// Fill out your copyright notice in the Description page of Project Settings.

#include "TextToFaceClient.h"
#include "TextToFaceLiveLinkSource.h"

#include "HttpModule.h"
#include "Interfaces/IHttpResponse.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Engine/World.h"
#include "Features/IModularFeatures.h"
#include "ILiveLinkClient.h"

DEFINE_LOG_CATEGORY_STATIC(LogTextToFace, Log, All);

UTextToFaceClient::UTextToFaceClient()
{
	PrimaryComponentTick.bCanEverTick = true;
}

// ---------------------------------------------------------------------------
// LiveLink source management
// ---------------------------------------------------------------------------

void UTextToFaceClient::InitLiveLinkSource()
{
	IModularFeatures& ModularFeatures = IModularFeatures::Get();
	if (!ModularFeatures.IsModularFeatureAvailable(ILiveLinkClient::ModularFeatureName))
	{
		UE_LOG(LogTextToFace, Warning, TEXT("LiveLink client not available — is the LiveLink plugin enabled?"));
		return;
	}

	LiveLinkSource = MakeShared<FTextToFaceLiveLinkSource>(LiveLinkSubjectName);

	ILiveLinkClient& Client = ModularFeatures.GetModularFeature<ILiveLinkClient>(
		ILiveLinkClient::ModularFeatureName);
	Client.AddSource(LiveLinkSource);

	UE_LOG(LogTextToFace, Log, TEXT("LiveLink source created, subject='%s'"),
		*LiveLinkSubjectName.ToString());
}

void UTextToFaceClient::PushLiveLinkFrame()
{
	if (!LiveLinkSource.IsValid() || !LiveLinkSource->IsReady()) return;

	FScopeLock Lock(&BlockMutex);
	if (!CurrentBlock.IsValid() || StartTimeSeconds < 0.0) return;
	if (CurrentBlock.ARKitFrames.Num() == 0) return;

	const UWorld* World = GetWorld();
	if (!World) return;

	const double Elapsed = World->GetTimeSeconds() - StartTimeSeconds;
	if (Elapsed < 0.0) return;

	const int32 LastFrame = CurrentBlock.NumFrames - 1;
	const double FrameF = Elapsed * (double)CurrentBlock.Fps;
	const int32 NumProps = CurrentBlock.ARKitPropertyNames.Num();

	TArray<float> Values;
	Values.SetNum(NumProps);

	if (FrameF >= (double)LastFrame)
	{
		// Clip finished — push zeros to clear additive layer, then stop
		StartTimeSeconds = -1.0;
		TArray<float> Zeros;
		Zeros.SetNumZeroed(NumProps);
		LiveLinkSource->PushFrameData(Zeros);
		return;
	}
	else
	{
		// Linearly interpolate between frames
		const int32 F0 = (int32)FrameF;
		const int32 F1 = F0 + 1;
		const float Alpha = (float)(FrameF - (double)F0);

		const TArray<float>& V0 = CurrentBlock.ARKitFrames[F0];
		const TArray<float>& V1 = CurrentBlock.ARKitFrames[F1];
		for (int32 i = 0; i < NumProps; ++i)
		{
			Values[i] = FMath::Lerp(V0[i], V1[i], Alpha);
		}
	}

	LiveLinkSource->PushFrameData(Values);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void UTextToFaceClient::BeginPlay()
{
	Super::BeginPlay();
	InitLiveLinkSource();
}

void UTextToFaceClient::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (LiveLinkSource.IsValid())
	{
		LiveLinkSource->RequestSourceShutdown();
		LiveLinkSource.Reset();
	}
	Super::EndPlay(EndPlayReason);
}

void UTextToFaceClient::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	PushLiveLinkFrame();
}

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

void UTextToFaceClient::GenerateExpression(const FString& Prompt, int32 Frames, int32 Fps)
{
	if (Prompt.IsEmpty())
	{
		UE_LOG(LogTextToFace, Warning, TEXT("GenerateExpression: empty prompt"));
		return;
	}

	const int32 UseFrames = (Frames > 0) ? Frames : DefaultFrames;
	const int32 UseFps    = (Fps    > 0) ? Fps    : DefaultFps;

	const TSharedRef<FJsonObject> Body = MakeShared<FJsonObject>();
	Body->SetStringField(TEXT("prompt"), Prompt);
	Body->SetNumberField(TEXT("frames"), UseFrames);
	Body->SetNumberField(TEXT("fps"),    UseFps);

	FString BodyString;
	const TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&BodyString);
	FJsonSerializer::Serialize(Body, Writer);

	const TSharedRef<IHttpRequest> Req = FHttpModule::Get().CreateRequest();
	Req->SetVerb(TEXT("POST"));
	Req->SetURL(ServerUrl + TEXT("/generate"));
	Req->SetHeader(TEXT("Content-Type"), TEXT("application/json"));
	Req->SetContentAsString(BodyString);
	Req->SetTimeout(RequestTimeoutSeconds);

	Req->OnProcessRequestComplete().BindUObject(
		this, &UTextToFaceClient::HandleHttpResponse, Prompt);

	if (!Req->ProcessRequest())
	{
		UE_LOG(LogTextToFace, Warning, TEXT("ProcessRequest failed for \"%s\""), *Prompt);
		FTextToFaceCurveBlock Failed;
		Failed.Prompt = Prompt;
		OnGenerationComplete.Broadcast(Failed);
		return;
	}

	UE_LOG(LogTextToFace, Log, TEXT("sent: \"%s\" (%d frames @ %d fps)"), *Prompt, UseFrames, UseFps);
}

// ---------------------------------------------------------------------------
// Response
// ---------------------------------------------------------------------------

void UTextToFaceClient::HandleHttpResponse(
	FHttpRequestPtr Request, FHttpResponsePtr Response,
	bool bConnectedSuccessfully, FString Prompt)
{
	FTextToFaceCurveBlock Block;
	Block.Prompt = Prompt;

	if (!bConnectedSuccessfully || !Response.IsValid())
	{
		UE_LOG(LogTextToFace, Warning, TEXT("request failed for \"%s\""), *Prompt);
		OnGenerationComplete.Broadcast(Block);
		return;
	}
	if (Response->GetResponseCode() != 200)
	{
		UE_LOG(LogTextToFace, Warning, TEXT("server returned %d for \"%s\""),
			Response->GetResponseCode(), *Prompt);
		OnGenerationComplete.Broadcast(Block);
		return;
	}
	if (!ParseResponseJson(Response->GetContentAsString(), Block))
	{
		UE_LOG(LogTextToFace, Warning, TEXT("parse failed for \"%s\""), *Prompt);
		OnGenerationComplete.Broadcast(Block);
		return;
	}

	Block.bSuccess = true;

	{
		FScopeLock Lock(&BlockMutex);
		CurrentBlock = Block;
		if (bAutoPlayOnReceive)
		{
			if (const UWorld* World = GetWorld())
			{
				StartTimeSeconds = World->GetTimeSeconds();
			}
		}
	}

	// Push static data to LiveLink (property names, once per generation)
	if (LiveLinkSource.IsValid() && LiveLinkSource->IsReady() && Block.ARKitPropertyNames.Num() > 0)
	{
		LiveLinkSource->PushStaticData(Block.ARKitPropertyNames);
	}

	UE_LOG(LogTextToFace, Log,
		TEXT("received \"%s\": %d ARKit channels x %d frames, %.1fms"),
		*Prompt, Block.ARKitPropertyNames.Num(), Block.NumFrames, Block.GenerationMs);

	OnGenerationComplete.Broadcast(Block);
}

bool UTextToFaceClient::ParseResponseJson(
	const FString& JsonString, FTextToFaceCurveBlock& OutBlock) const
{
	TSharedPtr<FJsonObject> Root;
	const TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid()) return false;

	double D = 0.0;
	Root->TryGetNumberField(TEXT("duration"),      D); OutBlock.Duration     = (float)D;
	Root->TryGetNumberField(TEXT("fps"),           D); OutBlock.Fps          = (int32)D;
	Root->TryGetNumberField(TEXT("n_frames"),      D); OutBlock.NumFrames    = (int32)D;
	Root->TryGetNumberField(TEXT("generation_ms"), D); OutBlock.GenerationMs = (float)D;

	// --- Parse arkit_raw (for LiveLink path) ---
	const TSharedPtr<FJsonObject>* ARKitObj = nullptr;
	if (Root->TryGetObjectField(TEXT("arkit_raw"), ARKitObj) && ARKitObj && ARKitObj->IsValid())
	{
		// Collect property names and per-property arrays
		TArray<FName> Names;
		TArray<TArray<float>> PerPropertyArrays;

		for (const auto& Pair : (*ARKitObj)->Values)
		{
			const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
			if (!Pair.Value->TryGetArray(Arr) || !Arr) continue;

			TArray<float> Vals;
			Vals.Reserve(Arr->Num());
			for (const TSharedPtr<FJsonValue>& V : *Arr)
			{
				Vals.Add((float)V->AsNumber());
			}
			Names.Add(FName(*Pair.Key));
			PerPropertyArrays.Add(MoveTemp(Vals));
		}

		OutBlock.ARKitPropertyNames = MoveTemp(Names);

		// Transpose: per-property arrays → per-frame arrays
		const int32 NumProps = OutBlock.ARKitPropertyNames.Num();
		const int32 NumFrames = (NumProps > 0) ? PerPropertyArrays[0].Num() : 0;
		OutBlock.ARKitFrames.SetNum(NumFrames);
		for (int32 F = 0; F < NumFrames; ++F)
		{
			OutBlock.ARKitFrames[F].SetNum(NumProps);
			for (int32 P = 0; P < NumProps; ++P)
			{
				OutBlock.ARKitFrames[F][P] = PerPropertyArrays[P][F];
			}
		}
	}

	// --- Parse curves (for legacy AnimNode path) ---
	const TSharedPtr<FJsonObject>* CurvesObj = nullptr;
	if (Root->TryGetObjectField(TEXT("curves"), CurvesObj) && CurvesObj && CurvesObj->IsValid())
	{
		OutBlock.Curves.Empty();
		for (const auto& Pair : (*CurvesObj)->Values)
		{
			const TArray<TSharedPtr<FJsonValue>>* Arr = nullptr;
			if (!Pair.Value->TryGetArray(Arr) || !Arr) continue;
			TArray<float> Vals;
			Vals.Reserve(Arr->Num());
			for (const TSharedPtr<FJsonValue>& V : *Arr)
				Vals.Add((float)V->AsNumber());
			OutBlock.Curves.Add(FName(*Pair.Key), MoveTemp(Vals));
		}
	}

	return OutBlock.NumFrames > 0;
}

// ---------------------------------------------------------------------------
// Playback controls
// ---------------------------------------------------------------------------

void UTextToFaceClient::PlayLastBlock()
{
	FScopeLock Lock(&BlockMutex);
	if (!CurrentBlock.IsValid()) return;
	if (const UWorld* World = GetWorld())
		StartTimeSeconds = World->GetTimeSeconds();
}

void UTextToFaceClient::Stop()
{
	FScopeLock Lock(&BlockMutex);
	StartTimeSeconds = -1.0;
}

bool UTextToFaceClient::IsPlaying() const
{
	FScopeLock Lock(&BlockMutex);
	return CurrentBlock.IsValid() && StartTimeSeconds >= 0.0;
}

bool UTextToFaceClient::SampleCurrentCurves(
	TMap<FName, float>& OutCurves, float& OutProgress01) const
{
	OutCurves.Reset();
	OutProgress01 = 0.f;

	FScopeLock Lock(&BlockMutex);
	if (!CurrentBlock.IsValid() || StartTimeSeconds < 0.0) return false;
	if (CurrentBlock.Curves.Num() == 0) return false;

	const UWorld* World = GetWorld();
	if (!World) return false;

	const double Elapsed = World->GetTimeSeconds() - StartTimeSeconds;
	if (Elapsed < 0.0) return false;

	const int32 LastFrame = CurrentBlock.NumFrames - 1;
	const double FrameF = Elapsed * (double)CurrentBlock.Fps;

	OutCurves.Reserve(CurrentBlock.Curves.Num());

	if (FrameF >= (double)LastFrame)
	{
		for (const auto& Pair : CurrentBlock.Curves)
			OutCurves.Add(Pair.Key, Pair.Value[LastFrame]);
		OutProgress01 = 1.f;
		return true;
	}

	const int32 F0 = (int32)FrameF;
	const int32 F1 = F0 + 1;
	const float Alpha = (float)(FrameF - (double)F0);

	for (const auto& Pair : CurrentBlock.Curves)
		OutCurves.Add(Pair.Key, FMath::Lerp(Pair.Value[F0], Pair.Value[F1], Alpha));

	OutProgress01 = CurrentBlock.Duration > 0.f ? (float)(Elapsed / CurrentBlock.Duration) : 0.f;
	return true;
}
