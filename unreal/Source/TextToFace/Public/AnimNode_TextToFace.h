// Custom AnimNode that reads ctrl_expressions_* curves from a
// UTextToFaceClient component on the owning actor and writes them
// into the output pose each frame.

#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimNodeBase.h"

#if WITH_EDITORONLY_DATA
#include "AnimGraphNode_Base.h"
#endif

#include "AnimNode_TextToFace.generated.h"

class UTextToFaceClient;

USTRUCT(BlueprintInternalUseOnly)
struct MH_EXPRESS_TEST_API FAnimNode_TextToFace : public FAnimNode_Base
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Links")
	FPoseLink SourcePose;

	virtual void Initialize_AnyThread(const FAnimationInitializeContext& Context) override;
	virtual void CacheBones_AnyThread(const FAnimationCacheBonesContext& Context) override;
	virtual void Update_AnyThread(const FAnimationUpdateContext& Context) override;
	virtual void Evaluate_AnyThread(FPoseContext& Output) override;

private:
	TWeakObjectPtr<UTextToFaceClient> CachedClient;
	TMap<FName, float> SampledCurves;
	float SampledProgress = 0.f;

	// Cached bone index for direct head bone rotation (bypasses ControlRig)
	FCompactPoseBoneIndex HeadBoneIndex = FCompactPoseBoneIndex(INDEX_NONE);
};

#if WITH_EDITORONLY_DATA

UCLASS()
class UAnimGraphNode_TextToFace : public UAnimGraphNode_Base
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, Category = "Settings")
	FAnimNode_TextToFace Node;

	virtual FText GetNodeTitle(ENodeTitleType::Type TitleType) const override;
	virtual FText GetTooltipText() const override;
	virtual FString GetNodeCategory() const override;
};

#endif
