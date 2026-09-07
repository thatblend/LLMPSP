#include "falcon_h1.h"

#include <stdio.h>
#include <string.h>

typedef struct {
    const char *text;
    int expected[16];
    int count;
} Case;

int main(int argc, char **argv) {
    static const Case cases[] = {
        {"Hello world!", {12698, 2388, 524}, 3},
        {" A test 123.", {854, 1673, 731, 540, 541, 542, 537}, 7},
        {"What is 2+2?\n", {4662, 860, 731, 541, 534, 541, 554, 709}, 8},
        {"  trailing  ", {731, 24405, 767}, 3},
        {"hello-world_test", {13722, 536, 9716, 586, 1297}, 5},
        {"user\n", {2101, 709}, 2},
        {"assistant\n", {963, 10259, 709}, 3}
    };
    static const int chat_expected[] = {
        227, 2101, 709, 12698, 893, 2668,
        228, 709, 227, 963, 10259, 709
    };
    FalconModel model;
    char error[256];
    int output[128], failed = 0;
    size_t c, i;
    if (argc != 2) {
        fprintf(stderr, "usage: %s model.fhq4\n", argv[0]);
        return 2;
    }
    if (!falcon_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    for (c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
        int count = falcon_tokenize(&model.tokenizer, cases[c].text,
                                    output, (int)(sizeof(output) / sizeof(output[0])));
        int match = count == cases[c].count;
        for (i = 0; match && i < (size_t)count; ++i)
            if (output[i] != cases[c].expected[i]) match = 0;
        printf("%s: %s", cases[c].text, match ? "ok" : "FAIL");
        if (!match) {
            printf(" [");
            for (i = 0; i < (size_t)(count > 0 ? count : 0); ++i)
                printf("%s%d", i ? "," : "", output[i]);
            printf("]");
            failed = 1;
        }
        printf("\n");
    }
    {
        int count = falcon_build_chat_prompt(&model, "Hello PSP",
                                             output, (int)(sizeof(output) / sizeof(output[0])));
        int match = count == (int)(sizeof(chat_expected) / sizeof(chat_expected[0]));
        for (i = 0; match && i < (size_t)count; ++i)
            if (output[i] != chat_expected[i]) match = 0;
        printf("chat template: %s\n", match ? "ok" : "FAIL");
        if (!match) failed = 1;
    }
    falcon_model_close(&model);
    return failed;
}
