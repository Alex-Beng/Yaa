

#include <dinput.h>
#include <Windows.h>
#include <cstdio>
#include <ctime>

int main()
{
	IDirectInput8W* idi8;
	IDirectInputDevice8W* dev;
	DIMOUSESTATE2 s;
	HINSTANCE h = GetModuleHandle(NULL);
	DirectInput8Create(h, 0x0800, IID_IDirectInput8, (void**)&idi8, NULL);
	if (!SUCCEEDED(idi8->CreateDevice(GUID_SysMouse, &dev, NULL))) {
		return 1;
	}
	dev->SetDataFormat(&c_dfDIMouse2);
	dev->SetCooperativeLevel(NULL, DISCL_BACKGROUND | DISCL_NONEXCLUSIVE);
	while (true) {
		dev->Acquire();
		dev->GetDeviceState(sizeof(s), &s);
		printf("%d %d %d %d\n", s.lX, s.lY, s.rgbButtons[0], s.rgbButtons[1]);
		fflush(stdout);
		Sleep(CLOCKS_PER_SEC);
	}
}