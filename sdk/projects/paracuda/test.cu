#include <cstdio>
#include <ctime>

#define USE_NVIDIA_SCAN
#define REPEATS 10

#include "arrayprint.h"
#include "paracuda.h"
#include "kernel.cu"
#include "split.h"
#include "copy.h"
#include "radix.h"
#include "cpu_radix.h"
#include "cpu_split.h"
#include "cpu_map.h"
#include "cpu_scan.h"
#include "nvidia_scan.h"

void testCpuRadix(int* data, int* gold, int length, unsigned int timer)
{
    srand(time(0));
    for(int i = 0; i < length; i++) {
        data[i] = rand() % 1000;
    }
    
    int* temp = (int*) malloc(length * sizeof(int));
    cutilCheckError(cutStartTimer(timer));
    cpu_radix(data, temp, length);
    cutilCheckError(cutStopTimer(timer));
    free(temp);
    memcpy(gold, data, length * sizeof(int)); // only test speed
}

void testCpuSplit(int* data, int* gold, int length, unsigned int timer)
{
    srand(time(0));
    for(int i = 0; i < length; i++) {
        data[i] = rand() % 1000;
    }
    
    int* temp = (int*) malloc(length * sizeof(int));
    cutilCheckError(cutStartTimer(timer));
    cpu_split(data, temp, temp, length);
    cutilCheckError(cutStopTimer(timer));
    free(temp);
    memcpy(gold, data, length * sizeof(int)); // only test speed
}

void testCpuMap(int* data, int* gold, int length, unsigned int timer)
{
    srand(time(0));
    for(int i = 0; i < length; i++) {
        data[i] = rand() % 1000;
    }
    
    cutilCheckError(cutStartTimer(timer));
    cpu_map(data, length);
    cutilCheckError(cutStopTimer(timer));
    memcpy(gold, data, length * sizeof(int)); // only test speed
}

void testCpuScan(int* data, int* gold, int length, unsigned int timer)
{
    srand(time(0));
    for(int i = 0; i < length; i++) {
        data[i] = rand() % 1000;
    }
    
    cutilCheckError(cutStartTimer(timer));
    cpu_scan(data, length);
    cutilCheckError(cutStopTimer(timer));
    memcpy(gold, data, length * sizeof(int)); // only test speed
}

void testNvidiaScan(int* data, int* gold, int length, unsigned int timer)
{
    srand(time(0));
    for(int i = 0; i < length; i++) {
        data[i] = rand() % 1000;
    }

    int* device_data = PARACUDA_int_allocate_device(length);
    PARACUDA_int_copy_host_device(device_data, data, length);
    preallocBlockSums(length);
    cutilCheckError(cutStartTimer(timer));
    prescanArray(device_data, device_data, length);
    cutilCheckError(cutStopTimer(timer));
    deallocBlockSums();    
    PARACUDA_int_copy_device_host(data, device_data, length);
    PARACUDA_int_free_device(device_data);

    memcpy(gold, data, length * sizeof(int)); // only test speed
}

void testRadix(int* data, int* gold, int length, unsigned int timer)
{
    srand(time(0));
    for(int i = 0; i < length; i++) {
        data[i] = rand() % 1000;
    }
    
    for(int i = 0; i < length; i++) {
        gold[i] = data[i];
    }

    int* temp = (int*) malloc(length * sizeof(int));
    cpu_radix(gold, temp, length);
    free(temp);

    int* device_data = PARACUDA_int_allocate_device(length);
    PARACUDA_int_copy_host_device(device_data, data, length);
    cutilCheckError(cutStartTimer(timer));
    radix(device_data, length, PARACUDA_MAX_THREADS);
    cutilCheckError(cutStopTimer(timer));
    PARACUDA_int_copy_device_host(data, device_data, length);
    PARACUDA_int_free_device(device_data);
}

void testSplit(int* data, int* gold, int length, unsigned int timer)
{
    int* flag = PARACUDA_int_allocate_host(length);
    for(int i = 0; i < length; i++) {
        data[i] = i;
        flag[i] = i % 2 == 0;
    }
    int numZero = 0;
    int indexForZero = 0;
    for(int i = 0; i < length; i++) {
        if(flag[i] == 0) 
        numZero++;
    }
    for(int i = 0; i < length; i++) {
        if(flag[i] == 1) {
            gold[numZero] = data[i];
            numZero++;
        } else {
            gold[indexForZero] = data[i];
            indexForZero++;
        }
    } 
    int* d_data = PARACUDA_int_allocate_device(length);
    int* d_flags = PARACUDA_int_allocate_device(length);
    PARACUDA_int_copy_host_device(d_data, data, length);
    PARACUDA_int_copy_host_device(d_flags, flag, length);
    cutilCheckError(cutStartTimer(timer));
    split(d_data, d_flags, length);
    cutilCheckError(cutStopTimer(timer));
    PARACUDA_int_copy_device_host(data, d_data, length);
    PARACUDA_int_free_device(d_data);
    PARACUDA_int_free_device(d_flags);
    free(flag);
}

void testPermute(int* data, int* gold, int length, unsigned int timer)
{
    int* positions = (int*) malloc(length * sizeof(int));
    for(int i = 0; i < length; i++) {
      data[i] = i;
      positions[i] = length - i - 1;
    }
    int_permute(gold, gold, positions, length, PARACUDA_MAX_THREADS); // warmup
    for(int i = 0; i < length; i++) {
      gold[positions[i]] = data[i];
    }
    int* d_out = PARACUDA_int_allocate_device(length);
    int* d_data = PARACUDA_int_allocate_device(length);
    int* d_positions = PARACUDA_int_allocate_device(length);
    PARACUDA_int_copy_host_device(d_data, data, length);
    PARACUDA_int_copy_host_device(d_positions, positions, length);
    cutilCheckError(cutStartTimer(timer));
    PARACUDA_int_permute_run(d_out, d_data, d_positions, length, PARACUDA_MAX_THREADS);
    cutilCheckError(cutStopTimer(timer));
    PARACUDA_int_copy_device_host(data, d_out, length);
    PARACUDA_int_free_device(d_out);
    PARACUDA_int_free_device(d_data);
    PARACUDA_int_free_device(d_positions);
    free(positions);
}

void testScan(int* data, int* gold, int length, unsigned int timer)
{
    for(int i = 0; i < length; i++) {
        data[i] = i;
    }
    int sum = 0;
    for(int i = 0; i < length; i++) {
        gold[i] = sum;
        sum += data[i];
    }

    int* d_data = PARACUDA_int_allocate_device(length);
    PARACUDA_int_copy_host_device(d_data, data, length);

    cutilCheckError(cutStartTimer(timer));
    PARACUDA_plus_scan_run(0, d_data, length, PARACUDA_MAX_THREADS);
    cutilCheckError(cutStopTimer(timer));

    PARACUDA_int_copy_device_host(data, d_data, length);
    PARACUDA_int_free_device(d_data);
}

void testSegmentedScan(int* data, int* gold, int length, unsigned int timer)
{
    int* flags = (int*) calloc(length, sizeof(int));
    for(int i = 0; i < length; i++) {
        data[i] = i;
    }
    flags[length / 2] = 1;
    int sum = 0;
    for(int i = 0; i < length; i++) {
        if (flags[i]) sum = 0;
        gold[i] = sum;
        sum += data[i];
    }
    int* d_flags = PARACUDA_int_allocate_device(length);
    int* d_flags_copy = PARACUDA_int_allocate_device(length);
    int* d_data = PARACUDA_int_allocate_device(length);
    PARACUDA_int_copy_host_device(d_data, data, length);
    PARACUDA_int_copy_host_device(d_flags, flags, length);
    PARACUDA_int_copy_host_device(d_flags_copy, flags, length);
    cutilCheckError(cutStartTimer(timer));
    PARACUDA_segmented_plus_scan_run(0, d_data, d_flags_copy, d_flags, 0, length, PARACUDA_MAX_THREADS);
    cutilCheckError(cutStopTimer(timer));
    PARACUDA_int_copy_device_host(data, d_data, length);
    PARACUDA_int_free_device(d_flags);
    PARACUDA_int_free_device(d_flags_copy);
    PARACUDA_int_free_device(d_data);
    free(flags);
}

void testMap(int* data, int* gold, int length, unsigned int timer)
{
    for(int i = 0; i < length; i++) {
        data[i] = i % 2 == 0;
    }
    for(int i = 0; i < length; i++) {
        gold[i] = !data[i];
    }

    int* d_data = PARACUDA_int_allocate_device(length);
    PARACUDA_int_copy_host_device(d_data, data, length);

    cutilCheckError(cutStartTimer(timer));
    PARACUDA_negate_map_run(d_data, d_data, length, PARACUDA_MAX_THREADS);
    cutilCheckError(cutStopTimer(timer));

    PARACUDA_int_copy_device_host(data, d_data, length);
    PARACUDA_int_free_device(d_data);
}

void testCopy(int* data, int* gold, int length, unsigned int timer)
{
    int value;
    for(int i = 0; i < length; i++) {
        gold[i] = value;
    }

    int* d_data = PARACUDA_int_allocate_device(length);

    cutilCheckError(cutStartTimer(timer));
    PARACUDA_int_copy_run(d_data, value, length, PARACUDA_MAX_THREADS);
    cutilCheckError(cutStopTimer(timer));

    PARACUDA_int_copy_device_host(data, d_data, length);
    PARACUDA_int_free_device(d_data);
}

#define TEST(T, f, name, lengths, count, times) {\
    for(int a = 0; a < count; a++) {\
        unsigned int timer = 0;\
        cutilCheckError(cutCreateTimer(&timer));\
        PARACUDA_##T##_struct* data = PARACUDA_##T##_allocate_host(lengths[a]);\
        PARACUDA_##T##_struct* gold = PARACUDA_##T##_allocate_host(lengths[a]);\
        printf("Running %s for %d elements:\n", name, lengths[a]);\
        for(int h = 0; h < REPEATS; h++) f(data, gold, lengths[a], timer);\
        bool same = true;\
        for(int b = 0; b < lengths[a]; b++) {\
            int wrong = !PARACUDA_##T##_equal(data, b, gold, b);\
            if (b < 10 || (wrong && same)) {\
                /*printf("%d:\t%d %d\t%d %d\n", b,*/			\
                    /*data->x[b], data->y[b], gold->x[b], gold->y[b]);*/	\
                printf("%d:\t%d\t%d\n", b,			\
                    data[b], gold[b]);	\
                if(wrong) printf("^^^^^^^^^^^^^^^^^^^^\n");\
            }\
            same &= !wrong;\
        }\
        if(same) {\
            printf("They are the SAME! Processing time: %f ms, average: %f ms.\n\n", \
                cutGetTimerValue(timer), cutGetAverageTimerValue(timer));\
        } else {\
            printf("They are DIFFERENT! Processing time: %f ms, average: %f ms.\n\n", \
                cutGetTimerValue(timer), cutGetAverageTimerValue(timer));\
        }\
	if(!same) { printf("Errornous output, aborting test.\n"); exit(-1); }\
        times[a] = cutGetAverageTimerValue(timer);\
        cutilCheckError(cutDeleteTimer(timer));\
    }\
}\

void runTest(int argc, char** argv) {
    if(argc == 1) {
        printf("USAGE: scripts/makerun.sh paracuda <testname1> <testname2...>\n");
        printf("EXAMPLE: scripts/makerun.sh paracuda split\n");
        return;
    }
    //PARACUDA_INITIALIZE_CUDA(argc, argv);
    cutilCheckMsg("Test");
    int all = strcmp(argv[1], "all") == 0;
    int lengths[] = {32, 64, 128, 256, 512, 1024, 2 * 1024, 4 * 1024, 8 * 1024, 16 * 1024, 
        32 * 1024, 64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024,
        2 * 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024};
    int count = sizeof(lengths) / sizeof(*lengths);
    float times[argc - 1][count];
    for(int i = 1; i < argc; i++) {
        char* name = argv[i];
        float* local = times[i - 1];
        if(strcmp(name, "cpu-radix") == 0) TEST(int, testCpuRadix, name, lengths, count, local)
        else if(strcmp(name, "cpu-split") == 0) TEST(int, testCpuSplit, name, lengths, count, local)
        else if(strcmp(name, "cpu-map") == 0) TEST(int, testCpuMap, name, lengths, count, local)
        else if(strcmp(name, "cpu-scan") == 0) TEST(int, testCpuScan, name, lengths, count, local)
        else if(strcmp(name, "nvidia-scan") == 0) TEST(int, testNvidiaScan, name, lengths, count, local)
	else if(strcmp(name, "scan") == 0) TEST(int, testScan, name, lengths, count, local)
	else if(strcmp(name, "segmented-scan") == 0) TEST(int, testSegmentedScan, name, lengths, count, local)
	else if(strcmp(name, "permute") == 0) TEST(int, testPermute, name, lengths, count, local)
        else if(strcmp(name, "map") == 0) TEST(int, testMap, name, lengths, count, local)
        else if(strcmp(name, "copy") == 0) TEST(int, testCopy, name, lengths, count, local)
        else if(strcmp(name, "split") == 0) TEST(int, testSplit, name, lengths, count, local)
        else if(strcmp(name, "radix") == 0) TEST(int, testRadix, name, lengths, count, local)
	else 
	{
            printf("Unknown test: %s\n", name);
            exit(-1);
	}
    }
#ifdef USE_NVIDIA_SCAN
    printf("# PLOT: (first row: elements, remaining rows: times in milliseconds) (NVIDIAs scan)\n");
#else
    printf("# PLOT: (first row: elements, remaining rows: times in milliseconds) (own scan)\n");
#endif
    printf("name");
    for(int l = 0; l < count; l++) {
        printf(",%d", lengths[l]);
    }
    printf("\n");
    for(int j = 1; j < argc; j++) {
        printf("%s", argv[j]);
        for(int l = 0; l < count; l++) {
            printf(",%f", times[j - 1][l]);
	}
        printf("\n");
    }
    //PARACUDA_EXIT_KEYPRESS(argc, argv);
}
