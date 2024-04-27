// Fill out your copyright notice in the Description page of Project Settings.


#include "CaptureManager.h"
#include "../HandyUtils.h"
#include "Kismet/KismetRenderingLibrary.h"
#include "Kismet/KismetSystemLibrary.h"
#include "IImageWrapper.h"
#include "Camera/CameraComponent.h"
#include "IImageWrapperModule.h"
#include "RenderGraphResources.h"
#include "GameFramework/Character.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "ShaderParameterUtils.h"

// Sets default values
ACaptureManager::ACaptureManager()
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	EnableExport = false;
}

void ACaptureManager::InitialComps()
{
	if (LevelSequence)
	{
		ALevelSequenceActor* CurrentLevelSequenceActor = nullptr;
		LevelSequencePlayer = ULevelSequencePlayer::CreateLevelSequencePlayer(
			GetWorld(), LevelSequence, FMovieSceneSequencePlaybackSettings(), CurrentLevelSequenceActor);
		LOG_ONSCREEN(FColor::Green, TEXT("LevelSequencePlayer created."));
		if (LevelSequencePlayer)
		{
			EndFrame = LevelSequencePlayer->GetFrameDuration();
			CaptureFrameRate = LevelSequencePlayer->GetFrameRate().AsDecimal();
			LevelSequencePlayer->Play();
			LOG_ONSCREEN(FColor::Green,
			             FString::Printf(TEXT("Start Playing LevelSequence. Length: %d, Target: %.2lf"), EndFrame,
				             CaptureFrameRate));

			EndFrame = FMath::Min(EndFrame, StartFrame + TargetCaptureLength);
		}
	}
	else
	{
		LOG_ONSCREEN(FColor::White, TEXT("No LevelSequence set."));
	}
	InitialRTAndScc2ds(CaptureResource);
}

void ACaptureManager::InitialRTAndScc2ds(FCaptureResource& InCR)
{
	for (int i = 0; i < CaptureItems.Num(); i++)
	{
		auto* Mat = CaptureItems[i].PPMaterial;
		auto BufferName = CaptureItems[i].BufferName;
		if (GEngine) GEngine->AddOnScreenDebugMessage(-1, 2.f, FColor::Silver, BufferName + " Read");
		UTextureRenderTarget2D* rt = nullptr;
		{
			rt = CreateRenderTarget(TargetWidth, TargetHeight, *BufferName);
		}

		auto* scc2d = NewObject<USceneCaptureComponent2D>(this);
		scc2d->RegisterComponent();
		scc2d->AttachToComponent(RootComponent, FAttachmentTransformRules::SnapToTargetIncludingScale);
		scc2d->SetComponentTickEnabled(false);
		scc2d->bTickInEditor = false;
		scc2d->TextureTarget = rt;
		scc2d->bUseCustomProjectionMatrix = true;

		if (BufferName == "SceneColor")
		{
			scc2d->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
		}
		else if (BufferName == "SkyColor")
		{
			scc2d->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
			scc2d->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
			scc2d->ShowOnlyActors = SkyColorShowOnlyLists;
			for (AActor* Actor : SkyColorShowOnlyLists)
			{
				scc2d->ShowOnlyActorComponents(Actor);
			}
		}
		else if (BufferName == "SkyDepth")
		{
			scc2d->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
			scc2d->AddOrUpdateBlendable(Mat);
			scc2d->PrimitiveRenderMode = ESceneCapturePrimitiveRenderMode::PRM_UseShowOnlyList;
			scc2d->ShowOnlyActors = SkyColorShowOnlyLists;
			for (AActor* Actor : SkyColorShowOnlyLists)
			{
				scc2d->ShowOnlyActorComponents(Actor);
			}
		}
		else if (BufferName == "STColor")
		{
			scc2d->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
			scc2d->AddOrUpdateBlendable(Mat);
		}
		else if (BufferName == "SceneColorNoST")
		{
			scc2d->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
			scc2d->AddOrUpdateBlendable(Mat);
		}
		else if (BufferName == "SceneColorNoSTAlpha")
		{
			scc2d->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
			scc2d->AddOrUpdateBlendable(Mat);
		}
		else
		{
			scc2d->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
			scc2d->AddOrUpdateBlendable(Mat);
		}

		// scc2d->ShowFlags.SetTemporalAA(true);
		// scc2d->ShowFlags.SetDirectionalLights(false);
		// scc2d->ShowFlags.SetDirectLighting(false);
		scc2d->ShowFlags.SetAtmosphere(IsSetAtmosphere);
		scc2d->ShowFlags.SetVisualizeSkyAtmosphere(IsSetVisualizeSkyAtmosphere);
		scc2d->ShowFlags.SetFog(IsSetFog);
		scc2d->ShowFlags.SetVolumetricFog(IsSetVolumetricFog);
		scc2d->ShowFlags.SetVolumes(IsSetVolumes);
		scc2d->ShowFlags.SetBloom(IsSetBloom);

		scc2d->ShowFlags.SetMotionBlur(false);
		scc2d->ShowFlags.SetScreenSpaceReflections(true);
		scc2d->ShowFlags.SetEyeAdaptation(true);
		scc2d->ShowFlags.SetSeparateTranslucency(true);
		scc2d->ShowFlags.SetLensFlares(false);
		scc2d->ShowFlags.SetVignette(false);

		InCR.PostProcessSCC2Ds.Add(scc2d);
		InCR.PostProcessRTs.Add(rt);
		InCR.ExportedFrameCountArray.Add(0);
		TSharedPtr<TQueue<FRenderRequestLinearStruct*>> LQ(new TQueue<FRenderRequestLinearStruct*>());
		InCR.RenderRequestLinearQueues.Add(LQ);
		TSharedPtr<TQueue<FRenderRequestStruct*>> Q(new TQueue<FRenderRequestStruct*>());
		InCR.RenderRequestQueues.Add(Q);
	}
}

// Called when the game starts or when spawned
void ACaptureManager::BeginPlay()
{
	Super::BeginPlay();
	StartTimestamp = FDateTime::Now().ToUnixTimestamp();
	if (BasePath.Len() <= 0)
	{
		BasePath = FPaths::ConvertRelativePathToFull(FPaths::ProjectSavedDir()) + "Output/";
	}
	else
	{
		BasePath = BasePath + "/";
	}
	LOG_ONSCREEN(FColor::Green, FString::Printf(TEXT("BasePath: %s"), GetData(BasePath)));
	LOG_ONSCREEN(FColor::Green, TEXT("CaptureManager BeginPlay"));
	if (EnableExport)
	{
		EnableCapture = true;
	}
	if (EnableCapture)
	{
		InitialComps();
		AATempRenderTargets.Init(nullptr, TempRenderTargetNum);
		for (int i = 0; i < TempRenderTargetNum; ++i)
		{
			AATempRenderTargets[i] = CreateRenderTarget(TargetWidth, TargetHeight,
			                                            *(FString("TMP_AA_") + FString::FromInt(i)));
		}
	}

	GEngine->Exec(GetWorld(), ToCStr(FString::Printf(TEXT("r.ExposureOffset %f"), ExposureOffset)));
	GEngine->Exec(GetWorld(),TEXT("r.TemporalAA.Algorithm 1"));
	GEngine->Exec(GetWorld(), TEXT("r.SeparateTranslucency 1"));
	GEngine->Exec(GetWorld(), TEXT("r.BasePassOutputsVelocity 1"));
	GEngine->Exec(GetWorld(), TEXT("r.BasePassForceOutputsVelocity 1"));
	IConsoleManager::Get().FindConsoleVariable(TEXT("r.ForceLOD"))->Set(0);
}

UTextureRenderTarget2D* ACaptureManager::CreateRenderTarget(int width, int height, FName name)
{
	UTextureRenderTarget2D* render_target = dynamic_cast<UTextureRenderTarget2D*>(NewObject<UTextureRenderTarget2D>());
	render_target->RenderTargetFormat = ETextureRenderTargetFormat::RTF_RGBA32f;
	render_target->ClearColor = FLinearColor::Black;
	render_target->bAutoGenerateMips = false;

	render_target->InitAutoFormat(width, height);
	render_target->bForceLinearGamma = false;
	render_target->bUseLegacyGamma = false;
	render_target->UpdateResourceImmediate(true);
	return render_target;
}

// Called every frame
void ACaptureManager::Tick(float DeltaTime)
{
	CurrentTimestamp = FDateTime::Now().ToUnixTimestamp();
	Super::Tick(DeltaTime);
	// TestReadSurface(DeltaTime);
	// return;

	if (LevelSequencePlayer)
	{
		double CurrentTime = LevelSequencePlayer->GetCurrentTime().AsSeconds();
		double EndTime = LevelSequencePlayer->GetEndTime().AsSeconds();

		// LOG_ONSCREEN(FColor::Green,
		// FString::Printf(TEXT("LevelSequencePlayer, current: %.2f, end: %.2f"), CurrentTime, EndTime));
		if (EndFrame > 0 && FrameCount > EndFrame)
		{
			UKismetSystemLibrary::QuitGame(this, nullptr, EQuitPreference::Quit, false);
		}
	}
	if (FrameCount > StartFrame)
	{
		if (EnableCapture)
		{
			int LeftTimeSeconds = -1;
			if (EndFrame > 0)
			{
				LeftTimeSeconds = double(CurrentTimestamp - StartTimestamp) / (FrameCount + 1) * (EndFrame -
					FrameCount);
				LOG_ONSCREEN_FULL(-1, DeltaTime, FColor::Green,
				                  FString::Printf(TEXT("CaptureManager running at frame: %d, LeftTime: %ds"),
					                  ExportedFrameCount, LeftTimeSeconds));
			}
			else
			{
				LOG_ONSCREEN_FULL(-1, DeltaTime, FColor::Green,
				                  FString::Printf(TEXT("CaptureManager running at frame: %d"), ExportedFrameCount));
			}

			TestTickCapturePos(CaptureResource, DeltaTime);

			if (EnableExport)
			{
				TestReadSurface(CaptureResource, DeltaTime);
				TestWriteToDisk(CaptureResource, DeltaTime, "");
			}
			ExportedFrameCount++;
		}
	}
	LOG_ONSCREEN_FULL(-1, DeltaTime, FColor::Green,
	                  FString::Printf(TEXT("Time: %ds"), CurrentTimestamp - StartTimestamp));
	FrameCount++;
}

void ACaptureManager::TestTickCapturePos(FCaptureResource& InCR, float DeltaTime)
{
	for (auto* scc2d : InCR.PostProcessSCC2Ds)
	{
		scc2d->SetWorldTransform(
			GetWorld()->GetFirstPlayerController()->PlayerCameraManager->GetActorTransform());
	}
}

void ACaptureManager::CopyRenderTarget(UTextureRenderTarget2D* src, UTextureRenderTarget2D* dst)
{
	FRHITexture* rhi_src = src->GameThread_GetRenderTargetResource()->TextureRHI;
	FRHITexture* rhi_dst = dst->GameThread_GetRenderTargetResource()->TextureRHI;

	ENQUEUE_RENDER_COMMAND(Copy)([rhi_src, rhi_dst](FRHICommandListImmediate& RHICmdList)
	{
		RHICmdList.CopyToResolveTarget(rhi_src, rhi_dst, FResolveParams{});
	});
}

void ACaptureManager::TestReadSurface(FCaptureResource& InCR, float DeltaTime)
{
	AActor* CameraTarget = GetWorld()->GetFirstPlayerController()->PlayerCameraManager->ViewTarget.Target;
	FMatrix ProjectionMatrix;
	{
		UCameraComponent* camera = dynamic_cast<UCameraComponent*>(
			CameraTarget->GetComponentByClass(UCameraComponent::StaticClass()));
		float FOV = camera->FieldOfView * (float)PI / 360.0f;
		float const XAxisMultiplier = 1.0f;
		float const YAxisMultiplier = TargetWidth / (float)TargetHeight;
		ProjectionMatrix = FReversedZPerspectiveMatrix(
			FOV,
			FOV,
			XAxisMultiplier,
			YAxisMultiplier,
			GNearClippingPlane,
			GNearClippingPlane
		);
	}

	for (int i = 0; i < InCR.PostProcessSCC2Ds.Num(); ++i)
	{
		auto* scc2d = InCR.PostProcessSCC2Ds[i];
		FTextureRenderTargetResource* rtResource = scc2d->TextureTarget->GameThread_GetRenderTargetResource();
		auto& BufferName = CaptureItems[i].BufferName;
		FRenderRequestLinearStruct* renderRequest = new FRenderRequestLinearStruct();

		
		scc2d->CustomProjectionMatrix = ProjectionMatrix;
		scc2d->TickComponent(DeltaTime, LEVELTICK_All, nullptr);
		USceneCaptureComponent::UpdateDeferredCaptures(GetWorld()->Scene);
		FReadSurfaceDataFlags ReadSurfaceDataFlags = FReadSurfaceDataFlags(RCM_MinMax, CubeFace_MAX);
		FReadSurfaceContext readSurfaceContext = {
			rtResource,
			&(renderRequest->Image),
			FIntRect(0, 0, rtResource->GetSizeXY().X, rtResource->GetSizeXY().Y),
			ReadSurfaceDataFlags
		};
		ENQUEUE_RENDER_COMMAND(SceneDrawCompletion)(
			[readSurfaceContext](FRHICommandListImmediate& RHICmdList)
			{
				RHICmdList.ReadSurfaceData(
					readSurfaceContext.SrcRenderTarget->GetRenderTargetTexture(),
					readSurfaceContext.Rect,
					*readSurfaceContext.OutData,
					readSurfaceContext.Flags
				);
			});
		// Notifiy new task in RenderQueue
		InCR.RenderRequestLinearQueues[i]->Enqueue(renderRequest);
	}
}

void ACaptureManager::TestWriteToDisk(FCaptureResource& InCR, float DeltaTime, FString Postfix)
{
	LOG_ONSCREEN_FULL(-1, DeltaTime, FColor::Red,
	                  FString::Printf(TEXT("Len: %d"), InCR.RenderRequestLinearQueues.Num()));
	for (int i = 0; i < InCR.RenderRequestLinearQueues.Num(); i++)
	{
		auto& BufferName = CaptureItems[i].BufferName;
		if (true)
		{
			auto& Q = InCR.RenderRequestLinearQueues[i];
			if (!Q->IsEmpty())
			{
				FRenderRequestLinearStruct* nextRenderRequest = nullptr;
				Q->Peek(nextRenderRequest);
				if (nextRenderRequest)
				{
					nextRenderRequest->RenderFence.BeginFence();
					nextRenderRequest->RenderFence.Wait();
					// if (true || nextRenderRequest->RenderFence.IsFenceComplete())
					{
						IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<
							IImageWrapperModule>(
							FName("ImageWrapper"));
						const FString FileName = FString::Printf(TEXT("frame_%d"), InCR.ExportedFrameCountArray[i]);
						FString OutputName = BasePath + BufferName +
							Postfix + "/" + FileName + ".EXR";

						static TSharedPtr<IImageWrapper> EXRImageWrapper = ImageWrapperModule.
							CreateImageWrapper(EImageFormat::EXR);
						int ExportWidth = TargetWidth;
						int ExportHeight = TargetHeight;
						if (BufferName == "WorldToClip")
						{
							ExportWidth = ExportHeight = 8;
						}
						EXRImageWrapper->SetRaw(nextRenderRequest->Image.GetData(),
						                        nextRenderRequest->Image.GetAllocatedSize(), ExportWidth, ExportHeight,
						                        ERGBFormat::RGBA, 32);
						const TArray64<uint8>& Data = EXRImageWrapper->GetCompressed(100);
						// const TArray64<uint8>& Data = EXRImageWrapper->GetCompressed((int32)EImageCompressionQuality::Uncompressed);

						FFileHelper::SaveArrayToFile(Data, *OutputName);

						InCR.RenderRequestLinearQueues[i]->Pop();
						delete nextRenderRequest;

						if (i == 0)
						{
							LOG_ONSCREEN_FULL(-1, DeltaTime, FColor::Yellow,
							                  FString::Printf(TEXT("size: %llu"), nextRenderRequest->Image.
								                  GetAllocatedSize(
								                  )));
							LOG_ONSCREEN_FULL(-1, DeltaTime, FColor::Yellow, OutputName);
						}

						InCR.ExportedFrameCountArray[i] += 1;
					}
				}
				Q->Pop();
			}
		}
	}
}
