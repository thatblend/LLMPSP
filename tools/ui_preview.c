/* Host preview of the PSP screen. Renders the same character grid the
 * PSP frontend blits, inside a border, so layout can be checked without
 * hardware. Also verifies that no row overflows the grid and that the
 * pixel row positions fit the 272-line panel without overlapping. */
#include "falcon_ui.h"

#include <stdio.h>
#include <string.h>

static void show(const char *caption, const FalconUiState *state) {
    char grid[UI_ROWS][UI_COLS + 1];
    int row, bad = 0;
    falcon_ui_compose(state, grid);
    printf("\n%s\n+", caption);
    for (row = 0; row < UI_COLS; ++row) putchar('-');
    printf("+\n");
    for (row = 0; row < UI_ROWS; ++row) {
        size_t length = strlen(grid[row]);
        printf("|%s|\n", grid[row]);
        if (length != UI_COLS) {
            printf("  !! row %d length %u\n", row, (unsigned)length);
            bad = 1;
        }
    }
    printf("+");
    for (row = 0; row < UI_COLS; ++row) putchar('-');
    printf("+\n");
    if (bad) printf("LAYOUT ERROR\n");
}

/* Rows are blitted at explicit pixel positions rather than on the font's
 * 8-pixel pitch, so check that they neither overlap nor run off the
 * bottom of the panel. */
static int check_pixel_rows(void) {
    int row, bad = 0, previous = -UI_FONT_HEIGHT;
    printf("\nrow pixel layout (glyphs %d tall, panel %d lines):\n",
           UI_FONT_HEIGHT, UI_SCREEN_HEIGHT);
    for (row = 0; row < UI_ROWS; ++row) {
        int y = falcon_ui_row_y(row);
        printf("  row %2d  y %3d..%3d  leading %d\n",
               row, y, y + UI_FONT_HEIGHT - 1, y - previous - UI_FONT_HEIGHT);
        if (y < previous + UI_FONT_HEIGHT) {
            printf("  !! row %d overlaps the row above it\n", row);
            bad = 1;
        }
        if (y + UI_FONT_HEIGHT > UI_SCREEN_HEIGHT) {
            printf("  !! row %d runs past the bottom of the screen\n", row);
            bad = 1;
        }
        previous = y;
    }
    if (bad) printf("PIXEL LAYOUT ERROR\n");
    return bad;
}

int main(void) {
    FalconUiState state;
    int bad;
    memset(&state, 0, sizeof(state));
    state.context_total = 512;
    bad = check_pixel_rows();

    /* 1. Fresh start, nothing typed yet. */
    falcon_ui_reset();
    falcon_ui_set_draft("");
    state.status = "Ready. 44 MiB cached, 20 of 24 layers in RAM.";
    show("=== 1. just launched ===", &state);

    /* 2. Typing: the message appears straight away as You:. */
    falcon_ui_set_draft("what is the capital of czechia");
    state.cursor = 18;
    state.status = "Ready. Press START to send.";
    show("=== 2. typing (appears once, at the top) ===", &state);

    /* 3. Sent: draft frozen, reply streaming in underneath. */
    falcon_ui_commit_draft("what is the capital of czechia");
    falcon_ui_append_str("The capital of Czechia is Prague, a city on the");
    state.busy = 1;
    state.context_used = 41;
    state.status = "Prompt processing... 100%";
    show("=== 3. generating ===", &state);

    /* 4. Reply done, ready for the next message. */
    falcon_ui_append_str(" Vltava river known for its historic old town.");
    falcon_ui_end_turn();
    falcon_ui_set_draft("");
    state.busy = 0;
    state.context_used = 63;
    state.status = "Done: 21 words in 44 s (2.1 s each).";
    show("=== 4. reply finished ===", &state);

    /* 5. A few turns in, scrolled to the newest text. */
    falcon_ui_commit_draft("how big is it");
    falcon_ui_append_str("Prague has roughly 1.3 million residents, making "
                         "it the largest city in the country by a wide "
                         "margin.");
    falcon_ui_end_turn();
    falcon_ui_commit_draft("what language do they speak there");
    falcon_ui_append_str("The official language is Czech, a West Slavic "
                         "language written with the Latin alphabet. Many "
                         "younger people also speak English.");
    falcon_ui_end_turn();
    falcon_ui_set_draft("thanks");
    state.cursor = 41;
    state.context_used = 168;
    state.scroll = falcon_ui_max_scroll();
    state.status = "Ready. Press START to send.";
    show("=== 5. several turns, scrolled to end ===", &state);

    /* 6. Scrolled back with shift held. */
    state.scroll = 0;
    state.upper = 1;
    state.status = "Scrolled up. Stick down returns to the newest reply.";
    show("=== 6. scrolled up, upper case ===", &state);

    printf("\nlines wrapped: %d, max scroll: %d\n",
           falcon_ui_line_count(), falcon_ui_max_scroll());
    return bad;
}
