// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Components/SkeletalMeshComponent.h"
#include "Interfaces/IHttpRequest.h"
#include "TextToFaceClient.generated.h"

class FTextToFaceLiveLinkSource;

/**
 * One generated expression block — metadata plus per-channel per-frame values.
 */
USTRUCT(BlueprintType)
struct MH_EXPRESS_TEST_API FTextToFaceCurveBlock
{
	GENERATED_BODY()

	UPROPERTY(BlueprintReadOnly, Category = "TextToFace")
	FString Prompt;

	UPROPERTY(BlueprintReadOnly, Category = "TextToFace")
	float Duration = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "TextToFace")
	int32 Fps = 60;

	UPROPERTY(BlueprintReadOnly, Category = "TextToFace")
	int32 NumFrames = 0;

	UPROPERTY(BlueprintReadOnly, Category = "TextToFace")
	float GenerationMs = 0.f;

	UPROPERTY(BlueprintReadOnly, Category = "TextToFace")
	bool bSuccess = false;

	// --- LiveLink path: raw ARKit channel data ---
	// Property names (61 ARKit channels: 52 blendshapes + 9 rotations)
	TArray<FName> ARKitPropertyNames;
	// Per-frame values: outer = frames, inner = 61 values per frame
	TArray<TArray<float>> ARKitFrames;

	// --- Legacy AnimNode path: ctrl_expressions_* mapped curves ---
	TMap<FName, TArray<float>> Curves;

	bool IsValid() const { return bSuccess && NumFrames > 0; }
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
	FTextToFaceGenerationComplete, const FTextToFaceCurveBlock&, CurveBlock);

/**
 * Drives MetaHuman facial animation from text prompts via a Python sidecar.
 *
 * Uses LiveLink to publish ARKit-format face data that the MetaHuman
 * consumes through its standard LiveLink face pipeline (ABP_MH_LiveLink).
 * This handles expressions, head rotation, AND eye gaze natively —
 * the same path the iPhone Live Link Face app uses.
 *
 * Setup:
 *   1. Add this component to a MetaHuman actor
 *   2. Copy ABP_MH_LiveLink to your project and set it as the Face's Anim Class
 *      (Animation Mode = Use Animation Blueprint)
 *   3. In ABP_MH_LiveLink, set the LiveLink Subject Name to match
 *      this component's LiveLinkSubjectName (default: "TextToFace")
 *   4. Call GenerateExpression("A person looks surprised")
 */
UCLASS(ClassGroup = (TextToFace), meta = (BlueprintSpawnableComponent))
class MH_EXPRESS_TEST_API UTextToFaceClient : public UActorComponent
{
	GENERATED_BODY()

public:
	UTextToFaceClient();

	// --- Config ------------------------------------------------------------

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TextToFace|Server")
	FString ServerUrl = TEXT("http://127.0.0.1:8765");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TextToFace|Server",
		meta = (ClampMin = "1", ClampMax = "480"))
	int32 DefaultFrames = 240;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TextToFace|Server",
		meta = (ClampMin = "1", ClampMax = "120"))
	int32 DefaultFps = 60;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TextToFace|Server",
		meta = (ClampMin = "0.5", ClampMax = "30.0"))
	float RequestTimeoutSeconds = 10.f;

	/** When true, PlayLastBlock() is called automatically on a successful generation. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TextToFace|Playback")
	bool bAutoPlayOnReceive = true;

	/** LiveLink subject name. Must match what the Face AnimBP listens for. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "TextToFace|LiveLink")
	FName LiveLinkSubjectName = FName(TEXT("TextToFace"));

	// --- Events ------------------------------------------------------------

	UPROPERTY(BlueprintAssignable, Category = "TextToFace")
	FTextToFaceGenerationComplete OnGenerationComplete;

	// --- API ---------------------------------------------------------------

	UFUNCTION(BlueprintCallable, Category = "TextToFace")
	void GenerateExpression(const FString& Prompt, int32 Frames = 0, int32 Fps = 0);

	UFUNCTION(BlueprintCallable, Category = "TextToFace")
	void PlayLastBlock();

	UFUNCTION(BlueprintCallable, Category = "TextToFace")
	void Stop();

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "TextToFace")
	bool IsPlaying() const;

	/** Legacy AnimNode path — sample mapped ctrl_expressions_* curves. */
	UFUNCTION(BlueprintCallable, Category = "TextToFace")
	bool SampleCurrentCurves(TMap<FName, float>& OutCurves, float& OutProgress01) const;

protected:
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
	void HandleHttpResponse(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bConnectedSuccessfully, FString Prompt);
	bool ParseResponseJson(const FString& JsonString, FTextToFaceCurveBlock& OutBlock) const;

	// LiveLink source (publishes ARKit face data each tick)
	TSharedPtr<FTextToFaceLiveLinkSource> LiveLinkSource;
	void InitLiveLinkSource();
	void PushLiveLinkFrame();

	mutable FCriticalSection BlockMutex;
	FTextToFaceCurveBlock CurrentBlock;
	double StartTimeSeconds = -1.0;
};
