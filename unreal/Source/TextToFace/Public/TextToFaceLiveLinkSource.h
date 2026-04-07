// LiveLink source that publishes ARKit face data from a TextToFaceClient.
// The MetaHuman consumes this through its standard LiveLink face pipeline,
// handling ARKit→rig mapping, head rotation, and eye gaze internally.

#pragma once

#include "CoreMinimal.h"
#include "ILiveLinkSource.h"
#include "ILiveLinkClient.h"

/**
 * Custom ILiveLinkSource that publishes pre-generated ARKit face data
 * (52 blendshapes + head/eye rotations) as a LiveLink subject.
 *
 * Created and owned by UTextToFaceClient. Registered with the LiveLink
 * client at BeginPlay, pushes frame data each tick during playback.
 */
class FTextToFaceLiveLinkSource : public ILiveLinkSource
{
public:
	FTextToFaceLiveLinkSource(const FName& InSubjectName);
	virtual ~FTextToFaceLiveLinkSource() = default;

	// --- ILiveLinkSource interface ---
	virtual void ReceiveClient(ILiveLinkClient* InClient, FGuid InSourceGuid) override;
	virtual void InitializeSettings(ULiveLinkSourceSettings* Settings) override {}
	virtual void Update() override {}
	virtual bool IsSourceStillValid() const override;
	virtual bool RequestSourceShutdown() override;
	virtual FText GetSourceType() const override;
	virtual FText GetSourceMachineName() const override;
	virtual FText GetSourceStatus() const override;

	// --- Our API ---

	/** Push the property name list (called once after registration). */
	void PushStaticData(const TArray<FName>& PropertyNames);

	/** Push one frame of property values (called each tick during playback). */
	void PushFrameData(const TArray<float>& PropertyValues);

	/** Check if the source is registered and ready to push data. */
	bool IsReady() const { return Client != nullptr && bIsValid; }

private:
	ILiveLinkClient* Client = nullptr;
	FGuid SourceGuid;
	FName SubjectName;
	bool bIsValid = true;
	bool bStaticDataPushed = false;
};
