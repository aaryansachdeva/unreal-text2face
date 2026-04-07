// LiveLink source implementation for TextToFace ARKit face data.

#include "TextToFaceLiveLinkSource.h"
#include "LiveLinkTypes.h"
#include "Roles/LiveLinkBasicRole.h"
#include "Roles/LiveLinkBasicTypes.h"

FTextToFaceLiveLinkSource::FTextToFaceLiveLinkSource(const FName& InSubjectName)
	: SubjectName(InSubjectName)
{
}

void FTextToFaceLiveLinkSource::ReceiveClient(ILiveLinkClient* InClient, FGuid InSourceGuid)
{
	Client = InClient;
	SourceGuid = InSourceGuid;
	UE_LOG(LogTemp, Log, TEXT("[LiveLinkSource] Registered with client, GUID=%s, Subject=%s"),
		*SourceGuid.ToString(), *SubjectName.ToString());
}

bool FTextToFaceLiveLinkSource::IsSourceStillValid() const
{
	return bIsValid;
}

bool FTextToFaceLiveLinkSource::RequestSourceShutdown()
{
	bIsValid = false;
	Client = nullptr;
	UE_LOG(LogTemp, Log, TEXT("[LiveLinkSource] Shutdown requested"));
	return true;
}

FText FTextToFaceLiveLinkSource::GetSourceType() const
{
	return FText::FromString(TEXT("TextToFace"));
}

FText FTextToFaceLiveLinkSource::GetSourceMachineName() const
{
	return FText::FromString(TEXT("localhost"));
}

FText FTextToFaceLiveLinkSource::GetSourceStatus() const
{
	return bIsValid
		? FText::FromString(TEXT("Active"))
		: FText::FromString(TEXT("Inactive"));
}

void FTextToFaceLiveLinkSource::PushStaticData(const TArray<FName>& PropertyNames)
{
	if (!Client || !bIsValid) return;

	FLiveLinkStaticDataStruct StaticDataStruct(FLiveLinkBaseStaticData::StaticStruct());
	FLiveLinkBaseStaticData* BaseData = StaticDataStruct.Cast<FLiveLinkBaseStaticData>();
	BaseData->PropertyNames = PropertyNames;

	const FLiveLinkSubjectKey SubjectKey(SourceGuid, SubjectName);
	Client->PushSubjectStaticData_AnyThread(
		SubjectKey,
		ULiveLinkBasicRole::StaticClass(),
		MoveTemp(StaticDataStruct)
	);

	bStaticDataPushed = true;
	UE_LOG(LogTemp, Log, TEXT("[LiveLinkSource] Pushed static data: %d properties for subject '%s'"),
		PropertyNames.Num(), *SubjectName.ToString());
}

void FTextToFaceLiveLinkSource::PushFrameData(const TArray<float>& PropertyValues)
{
	if (!Client || !bIsValid || !bStaticDataPushed) return;

	FLiveLinkFrameDataStruct FrameDataStruct(FLiveLinkBaseFrameData::StaticStruct());
	FLiveLinkBaseFrameData* FrameData = FrameDataStruct.Cast<FLiveLinkBaseFrameData>();
	FrameData->PropertyValues = PropertyValues;
	FrameData->WorldTime = FLiveLinkWorldTime(FPlatformTime::Seconds());

	const FLiveLinkSubjectKey SubjectKey(SourceGuid, SubjectName);
	Client->PushSubjectFrameData_AnyThread(
		SubjectKey,
		MoveTemp(FrameDataStruct)
	);
}
