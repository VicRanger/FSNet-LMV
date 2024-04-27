#pragma once

#define LOG_ONSCREEN_FULL(Key, Time, Color, String) if (GEngine) GEngine->AddOnScreenDebugMessage(Key, Time, Color, String)
#define LOG_ONSCREEN(Color, String) LOG_ONSCREEN_FULL(-1, 1.0f, Color, String)