// Custom AnimNode — reads curves from UTextToFaceClient, writes to pose.
// Also applies head bone rotation directly (bypassing ControlRig's HeadMovementIK
// which has known issues with yaw in UE 5.6).

#include "AnimNode_TextToFace.h"
#include "TextToFaceClient.h"
#include "Animation/AnimInstanceProxy.h"
#include "Components/SkeletalMeshComponent.h"
#include "BoneContainer.h"

void FAnimNode_TextToFace::Initialize_AnyThread(const FAnimationInitializeContext& Context)
{
	SourcePose.Initialize(Context);
	CachedClient.Reset();
	SampledCurves.Reset();
	HeadBoneIndex = FCompactPoseBoneIndex(INDEX_NONE);
}

void FAnimNode_TextToFace::CacheBones_AnyThread(const FAnimationCacheBonesContext& Context)
{
	SourcePose.CacheBones(Context);

	// Find the head bone for direct rotation — try multiple name candidates
	const FBoneContainer& RequiredBones = Context.AnimInstanceProxy->GetRequiredBones();
	HeadBoneIndex = FCompactPoseBoneIndex(INDEX_NONE);

	static const FName HeadCandidates[] = {
		FName(TEXT("head")), FName(TEXT("Head")),
		FName(TEXT("neck_02")), FName(TEXT("neck_01")),
	};
	for (const FName& Name : HeadCandidates)
	{
		const int32 MeshIndex = RequiredBones.GetPoseBoneIndexForBoneName(Name);
		if (MeshIndex != INDEX_NONE)
		{
			HeadBoneIndex = RequiredBones.MakeCompactPoseIndex(FMeshPoseBoneIndex(MeshIndex));
			UE_LOG(LogTemp, Warning, TEXT("[AnimNode] Head bone found: '%s' (mesh=%d, compact=%d)"),
				*Name.ToString(), MeshIndex, HeadBoneIndex.GetInt());
			break;
		}
	}
	// Head bone found status (one-time log only)
	static bool bLoggedBoneStatus = false;
	if (!bLoggedBoneStatus)
	{
		UE_LOG(LogTemp, Log, TEXT("[AnimNode] Head bone: %s"),
			HeadBoneIndex.GetInt() != INDEX_NONE ? TEXT("found") : TEXT("NOT found"));
		bLoggedBoneStatus = true;
	}
}

void FAnimNode_TextToFace::Update_AnyThread(const FAnimationUpdateContext& Context)
{
	SourcePose.Update(Context);

	if (!CachedClient.IsValid())
	{
		if (const FAnimInstanceProxy* Proxy = Context.AnimInstanceProxy)
		{
			if (const USkeletalMeshComponent* Mesh = Proxy->GetSkelMeshComponent())
			{
				if (const AActor* Owner = Mesh->GetOwner())
				{
					CachedClient = Owner->FindComponentByClass<UTextToFaceClient>();
				}
			}
		}
	}

	SampledCurves.Reset();
	SampledProgress = 0.f;
	if (CachedClient.IsValid())
	{
		CachedClient->SampleCurrentCurves(SampledCurves, SampledProgress);
	}
}

void FAnimNode_TextToFace::Evaluate_AnyThread(FPoseContext& Output)
{
	SourcePose.Evaluate(Output);

	// 1. Write expression curves (ctrl_expressions_*, consumed by PostProcess ControlRig)
	for (const TPair<FName, float>& Pair : SampledCurves)
	{
		Output.Curve.Set(Pair.Key, Pair.Value);
	}

	// 2. Apply head bone rotation directly (bypasses ControlRig HeadMovementIK)
	//    This survives PostProcess because the ControlRig only modifies face
	//    deformation bones, not the head bone itself.
	if (HeadBoneIndex.GetInt() != INDEX_NONE)
	{
		static const FName YawName(TEXT("HeadYaw"));
		static const FName PitchName(TEXT("HeadPitch"));
		static const FName RollName(TEXT("HeadRoll"));

		const float* YawPtr   = SampledCurves.Find(YawName);
		const float* PitchPtr = SampledCurves.Find(PitchName);
		const float* RollPtr  = SampledCurves.Find(RollName);

		if (YawPtr || PitchPtr || RollPtr)
		{
			const float Yaw   = YawPtr   ? *YawPtr   : 0.f;
			const float Pitch = PitchPtr ? *PitchPtr : 0.f;
			const float Roll  = RollPtr  ? *RollPtr  : 0.f;

			// Convert from radians to degrees for FRotator
			const FRotator HeadRotation(
				FMath::RadiansToDegrees(Pitch),
				FMath::RadiansToDegrees(Yaw),
				FMath::RadiansToDegrees(Roll)
			);

			FTransform& BoneTransform = Output.Pose[HeadBoneIndex];
			BoneTransform.SetRotation(
				FQuat(HeadRotation) * BoneTransform.GetRotation()
			);

		}
	}
}

#if WITH_EDITOR

FText UAnimGraphNode_TextToFace::GetNodeTitle(ENodeTitleType::Type TitleType) const
{
	return FText::FromString(TEXT("Text To Face Curves"));
}

FText UAnimGraphNode_TextToFace::GetTooltipText() const
{
	return FText::FromString(
		TEXT("Reads ctrl_expressions_* curves from a TextToFaceClient "
			 "component and writes them into the animation pose. "
			 "Also applies head bone rotation directly."));
}

FString UAnimGraphNode_TextToFace::GetNodeCategory() const
{
	return TEXT("TextToFace");
}

#endif
