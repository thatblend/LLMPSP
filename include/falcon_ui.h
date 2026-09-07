#ifndef FALCON_UI_H
#define FALCON_UI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The PSPSDK debug font is 7x8 pixels on a 480x272 panel, so its natural
 * grid is 68x34. Columns are limited to 67 so a full row never advances
 * the cursor, which would wrap it to the top of the screen.
 *
 * Rows, however, are NOT on the font's 8-pixel pitch: at that pitch the
 * wrapped chat has no leading at all and is tiring to read. Each row is
 * blitted at the pixel Y given by falcon_ui_row_y() below, which gives
 * chat lines 10 pixels of pitch instead of 8. The two extra pixels per
 * line are paid for by spacing the keyboard in pixels rather than with
 * blank grid rows, and by two chat lines (19 -> 17). */
#define UI_COLS         67
#define UI_ROWS         27
#define UI_FONT_WIDTH    7
#define UI_FONT_HEIGHT   8
#define UI_SCREEN_HEIGHT 272

/* Top: name, status, then the conversation including the message being
 * typed. Bottom: only the keyboard and controls. While a reply is being
 * generated the status moves to the bottom, where the controls are not
 * needed because typing is blocked anyway. */
#define UI_ROW_HEADER    0
#define UI_ROW_TOPSTATUS 1
#define UI_ROW_SEP_CHAT  2
#define UI_ROW_CHAT      3
#define UI_CHAT_ROWS     17
#define UI_ROW_SEP_KEYS  20
#define UI_ROW_KEYS      21
#define UI_KEY_ROWS      4
#define UI_KEY_COLUMNS   15
#define UI_KEY_ROW_STEP  1      /* keyboard rows are spaced in pixels */
#define UI_KEY_CELL      4      /* character cell width per key */
#define UI_ROW_HELP      25     /* two rows: controls, or busy + status */

#define UI_KEY_COUNT (UI_KEY_ROWS * UI_KEY_COLUMNS)

typedef struct {
    int cursor;              /* selected key index */
    int upper;               /* shift state */
    const char *status;      /* status line text */
    int context_used;
    int context_total;
    int scroll;              /* first visible chat line */
    int busy;                /* generation in progress */
} FalconUiState;

/* Conversation text. The message being typed is kept as a provisional
 * tail of the same buffer, so it wraps, scrolls and renders exactly like
 * committed history and can never appear twice on screen. */
void falcon_ui_reset(void);
void falcon_ui_set_draft(const char *text);      /* replace "You: ..." tail */
void falcon_ui_commit_draft(const char *text);   /* freeze it, open "AI: " */
void falcon_ui_append(const char *text, size_t length);  /* reply bytes */
void falcon_ui_append_str(const char *text);
void falcon_ui_end_turn(void);                   /* close the reply */

int falcon_ui_line_count(void);
int falcon_ui_max_scroll(void);
int falcon_ui_has_history(void);

const char *falcon_ui_keys(int upper);

/* Top pixel row of a grid row. Rows are 8 pixels tall and spaced with
 * extra leading, so this is not row * 8; see the layout note above.
 * Pure integer math, no PSP calls. */
int falcon_ui_row_y(int row);

/* Renders the whole screen into grid. Pure text: no PSP calls, no
 * floating point, safe to call between inference layers. */
void falcon_ui_compose(const FalconUiState *state,
                       char grid[UI_ROWS][UI_COLS + 1]);

#ifdef __cplusplus
}
#endif
#endif
