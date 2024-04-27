// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
// #include "MovieSceneSequencePlayer.h"
#include "LevelSequence/Public/LevelSequencePlayer.h"
#include "LevelSequence/Public/LevelSequence.h"
#include "Containers/Queue.h"
#include "CaptureManager.generated.h"

struct FReadSurfaceContext
{
	FRenderTarget* SrcRenderTarget;
	TArray<FLinearColor>* OutData;
	FIntRect Rect;
	FReadSurfaceDataFlags Flags;
};

USTRUCT()
struct FRenderRequestLinearStruct{
	GENERATED_BODY()
	TArray<FLinearColor> Image;
	FRenderCommandFence RenderFence;
	FRenderRequestLinearStruct(){
	}
};

USTRUCT()
struct FRenderRequestStruct{
	GENERATED_BODY()
	TArray<FColor> Image;
	FRenderCommandFence RenderFence;

};

USTRUCT()
struct FCaptureItem
{
	GENERATED_BODY()
	UPROPERTY(EditAnywhere)
	FString BufferName;
	UPROPERTY(EditAnywhere)
	UMaterial *PPMaterial;
};

USTRUCT()
struct FCaptureResource
{	
	GENERATED_BODY()
	UPROPERTY(VisibleAnywhere)
	TArray<UTextureRenderTarget2D*> PostProcessRTs;
	UPROPERTY(VisibleAnywhere)
	TArray<USceneCaptureComponent2D*> PostProcessSCC2Ds;
	UPROPERTY(VisibleAnywhere)
	TArray<int> ExportedFrameCountArray;
	TArray<TSharedPtr<TQueue<FRenderRequestLinearStruct*>>> RenderRequestLinearQueues;
	TArray<TSharedPtr<TQueue<FRenderRequestStruct*>>> RenderRequestQueues;
};

UCLASS()
class BUFFERREADER_API ACaptureManager : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	ACaptureManager();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	void InitialComps();
	void InitialRTAndScc2ds(FCaptureResource &InCR);
	// Creates an async task that will save the captured image to disk
	void RunAsyncImageSaveTask(TArray<uint8> Image, FString ImageName);

public:
	void CopyRenderTarget(UTextureRenderTarget2D* src, UTextureRenderTarget2D* dst);
	bool MultiSampleAdd(UTextureRenderTarget2D* sample, float InSampleScale, FRenderCommandFence Fence); 
	UTextureRenderTarget2D* CreateRenderTarget(int Width, int Height, FName Name);
	// Called every frame
	virtual void Tick(float DeltaTime) override;
	void TestTickCapturePos(FCaptureResource& InCR, float DeltaTime);
	// Read Surface
	void TestReadSurface(FCaptureResource& InCR, float DeltaTime);
	void TestWriteToDisk(FCaptureResource& InCR, float DeltaTime, FString Postfix);
public:
	const int TempRenderTargetNum = 2;
	int current_target_id = 0;
	UPROPERTY()
	TArray <UTextureRenderTarget2D*> AATempRenderTargets;
	UPROPERTY(EditAnywhere, Category = "1) Export Option")
	bool EnableCapture;
	UPROPERTY(EditAnywhere, Category = "1) Export Option")
	bool EnableExport;
	UPROPERTY(EditAnywhere, Category = "1) Export Option")
	FString BasePath;
	UPROPERTY(EditAnywhere, Category = "1) Export Option")
	uint32 TargetWidth = 1280;
	UPROPERTY(EditAnywhere, Category = "1) Export Option")
	uint32 TargetHeight = 720;
	UPROPERTY(EditAnywhere, Category = "1) Export Option")
	int StartFrame = 120;
	UPROPERTY(EditAnywhere, Category = "1) Export Option")
	int TargetCaptureLength = 1200;
	UPROPERTY(EditAnywhere, Category = "2) Buffer Option")
	TArray<FCaptureItem> CaptureItems;
	UPROPERTY(EditAnywhere, Category = "2) Buffer Option")
	TArray<AActor*> SkyColorShowOnlyLists;
	UPROPERTY(EditAnywhere)
	float ExposureOffset=0;
	UPROPERTY(EditAnywhere, Category = "PostProcessing Option")
	bool IsSetAtmosphere;
	UPROPERTY(EditAnywhere, Category = "PostProcessing Option")
	bool IsSetVisualizeSkyAtmosphere;
	UPROPERTY(EditAnywhere, Category = "PostProcessing Option")
	bool IsSetFog;
	UPROPERTY(EditAnywhere, Category = "PostProcessing Option")
	bool IsSetVolumetricFog;
	UPROPERTY(EditAnywhere, Category = "PostProcessing Option")
	bool IsSetVolumes;
	UPROPERTY(EditAnywhere, Category = "PostProcessing Option")
	bool IsSetBloom;
	UPROPERTY(VisibleAnywhere, Category = "Runtime Data")
	int EndFrame = -1;
	UPROPERTY(VisibleAnywhere, Category = "Runtime Data")
	double CaptureFrameRate = -1;
	UPROPERTY(VisibleAnywhere, Category = "Runtime Data")
	int StartTimestamp;
	UPROPERTY(VisibleAnywhere, Category = "Runtime Data")
	int CurrentTimestamp;
	UPROPERTY(VisibleAnywhere, Category = "Runtime Data")
	int FrameCount;
	UPROPERTY(VisibleAnywhere, Category = "Runtime Data")
	int ExportedFrameCount;
	UPROPERTY(EditAnywhere)
	ULevelSequence* LevelSequence;
	UPROPERTY(VisibleAnywhere, Category = "Runtime Data")
	ULevelSequencePlayer* LevelSequencePlayer;
	UPROPERTY(VisibleAnywhere)
	FCaptureResource CaptureResource;

};
