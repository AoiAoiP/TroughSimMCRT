CC = nvcc
PYTHON_INCLUDE = usr/include/python3.8
PYTHON_LIB_PATH = $(PYTHON_BASE_PATH)/lib
PYTHON_LIB = python3.8
MATPLOTLIB_INCLUDE = .

CFLAGS = -arch=sm_75 -O3 -std=c++11 # 编译选项
INCLUDES = -I/$(PYTHON_INCLUDE) # 包含选项

# 获取所有 .cu 文件
CU_FILES = $(wildcard *.cu)

# 生成对应的 .o 文件
OBJ_FILES = $(patsubst %.cu, %.o, $(CU_FILES))

# 主程序文件（不包含 CyliReflectors.cu）
MAIN_CU = Direct_MCRT_Simulation.cu
MAIN_OUT = Direct_MCRT_Simulation.out

# 默认目标
default: $(MAIN_OUT)

# 主程序链接规则
$(MAIN_OUT): $(OBJ_FILES)
	$(CC) $(CFLAGS) $(INCLUDES) $^ -o $@

# 编译每个 .cu 文件为 .o 文件
%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -dc $< -o $@

# 清理
clean:
	rm -f *.out *.o *.csv *.png