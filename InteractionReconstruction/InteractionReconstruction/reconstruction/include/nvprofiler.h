#include "nvToolsExt.h"
#include <vector>

class nvprofiler
{
	static std::vector<int> s_colorTable;
	static nvtxEventAttributes_t s_EventAttribute;
	static int s_colorCount;
	static bool s_flagEnable;

public:
	static void init(bool flagEnable = true, int colorCount = 8);
	static void start(const char* name, int colorId = 0);
	static void stop();
	static void enable();
	static void disable();
};