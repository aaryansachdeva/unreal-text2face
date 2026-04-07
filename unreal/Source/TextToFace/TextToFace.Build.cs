using UnrealBuildTool;

// NOTE: Rename this class to match YOUR project's module name.
// E.g. if your project is "MyProject", rename to: public class MyProject : ModuleRules
public class MH_Express_Test : ModuleRules
{
	public MH_Express_Test(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"HTTP",
			"Json",
			"AnimGraphRuntime",
			"LiveLinkInterface",  // ILiveLinkSource, ILiveLinkClient, FLiveLinkBaseStaticData
		});

		PrivateDependencyModuleNames.AddRange(new string[] { });

		if (Target.Type == TargetType.Editor)
		{
			PrivateDependencyModuleNames.AddRange(new string[]
			{
				"AnimGraph",
				"BlueprintGraph",
			});
		}
	}
}
