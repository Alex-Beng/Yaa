

#include <dinput.h>
#include <Windows.h>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <thread>

int main()
{
	IDirectInput8W* idi8;
	IDirectInputDevice8W* msDev;
	IDirectInputDevice8W* kbDev;
	DIMOUSESTATE2 msState;
	BYTE kbState[256];
	HINSTANCE h = GetModuleHandle(NULL);
	DirectInput8Create(h, 0x0800, IID_IDirectInput8, (void**)&idi8, NULL);
	// 创建鼠标设备
	if (!SUCCEEDED(idi8->CreateDevice(GUID_SysMouse, &msDev, NULL))) {
		return 1;
	}
	msDev->SetDataFormat(&c_dfDIMouse2);
	msDev->SetCooperativeLevel(NULL, DISCL_BACKGROUND | DISCL_NONEXCLUSIVE);

	// 创建键盘设备
	if (!SUCCEEDED(idi8->CreateDevice(GUID_SysKeyboard, &kbDev, NULL))) {
		return 1;
	}
	kbDev->SetDataFormat(&c_dfDIKeyboard);
	kbDev->SetCooperativeLevel(NULL, DISCL_BACKGROUND | DISCL_NONEXCLUSIVE);

	// 测试一下频率
	auto start = std::chrono::high_resolution_clock::now();
	int cnt = 0;

	while (true) {
		msDev->Acquire();
		msDev->GetDeviceState(sizeof(msState), &msState);
		if (msState.lX != 0 || msState.lY != 0 || msState.rgbButtons[0] != 0 || msState.rgbButtons[1] != 0) {
			printf("%d %d %d %d\n", msState.lX, msState.lY, msState.rgbButtons[0], msState.rgbButtons[1]);
		}

		kbDev->Acquire();
		kbDev->GetDeviceState(sizeof(kbState), kbState);
		for (int i = 0; i < 256; i++) {
			if (kbState[i] != 0) {
				printf("%d ", i);
			}
		}
		printf("\n");
		cnt += 1;
		auto end = std::chrono::high_resolution_clock::now();

		// 计算频率
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		if (duration.count() >= 1000) {
			printf("Frequency: %d\n", cnt);
			cnt = 0;
			start = end;
		}

		// fflush(stdout);
		Sleep(CLOCKS_PER_SEC/1000);
		// sleep for 0.1 ms
		// std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}