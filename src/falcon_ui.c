#include "falcon_ui.h"

#include <stdio.h>
#include <string.h>

#define TRANSCRIPT_BYTES 6144
#define MAX_WRAP_LINES 512

/* Wrapped continuation lines are indented so each turn reads as one
 * block. */
#define UI_WRAP_INDENT 2

/* transcript holds committed history followed by an optional provisional
 * "You: ..." tail for the message being typed. Everything past
 * committed_length is rewritten on each keystroke. */
static char transcript[TRANSCRIPT_BYTES];
static size_t transcript_length;
static size_t committed_length;

static unsigned short line_start[MAX_WRAP_LINES];
static unsigned char line_length[MAX_WRAP_LINES];
static unsigned char line_indent[MAX_WRAP_LINES];
static int line_total;
static int wrap_dirty = 1;

static const char lower_keys[] =
    "abcdefghijklmnopqrstuvwxyz0123456789.,!?'-_:;/()@#$%&*+=[]<>";
static const char upper_keys[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'-_:;/()@#$%&*+=[]<>";

const char *falcon_ui_keys(int upper) {
    return upper ? upper_keys : lower_keys;
}

/* Vertical layout, in pixels on the 272-line panel. Glyphs are 8 pixels
 * tall; anything above that is leading. Chat lines get 2 extra pixels
 * because at the bare font pitch the wrapped text runs together, and the
 * keyboard keeps visible separation from a 12-pixel pitch instead of the
 * blank grid rows it used before.
 *
 *   header 0..7, status 8..15, chat rule 18..25,
 *   17 chat lines 28..195 (pitch 10), keyboard rule 198..205,
 *   4 key rows 209..252 (pitch 12), 2 control rows 256..271.
 *
 * The last row therefore ends exactly on line 272; ui_preview checks it. */
#define UI_CHAT_PITCH  10
#define UI_KEY_PITCH   12
#define UI_Y_SEP_CHAT  18
#define UI_Y_CHAT      28
#define UI_Y_SEP_KEYS  198
#define UI_Y_KEYS      209
#define UI_Y_HELP      256

int falcon_ui_row_y(int row) {
    if (row < 0) row = 0;
    if (row >= UI_ROWS) row = UI_ROWS - 1;
    if (row <= UI_ROW_TOPSTATUS) return row * UI_FONT_HEIGHT;
    if (row < UI_ROW_CHAT) return UI_Y_SEP_CHAT;
    if (row < UI_ROW_SEP_KEYS)
        return UI_Y_CHAT + (row - UI_ROW_CHAT) * UI_CHAT_PITCH;
    if (row < UI_ROW_KEYS) return UI_Y_SEP_KEYS;
    if (row < UI_ROW_HELP)
        return UI_Y_KEYS + (row - UI_ROW_KEYS) * UI_KEY_PITCH;
    return UI_Y_HELP + (row - UI_ROW_HELP) * UI_FONT_HEIGHT;
}

void falcon_ui_reset(void) {
    transcript_length = 0;
    committed_length = 0;
    transcript[0] = '\0';
    line_total = 0;
    wrap_dirty = 1;
}

int falcon_ui_has_history(void) {
    return committed_length != 0;
}

/* Drops whole lines from the front so the newest text always survives. */
static void make_room(size_t needed) {
    size_t drop;
    if (transcript_length + needed < TRANSCRIPT_BYTES) return;
    drop = TRANSCRIPT_BYTES / 4;
    if (drop < needed) drop = needed;
    if (drop >= transcript_length) {
        transcript_length = 0;
        committed_length = 0;
        transcript[0] = '\0';
        return;
    }
    while (drop < transcript_length && transcript[drop] != '\n') ++drop;
    if (drop < transcript_length) ++drop;
    memmove(transcript, transcript + drop, transcript_length - drop);
    transcript_length -= drop;
    committed_length = committed_length > drop ? committed_length - drop : 0;
    transcript[transcript_length] = '\0';
}

static void raw_append(const char *text, size_t length) {
    size_t i;
    if (!text || !length) return;
    if (length >= TRANSCRIPT_BYTES - 1) {
        text += length - (TRANSCRIPT_BYTES - 2);
        length = TRANSCRIPT_BYTES - 2;
    }
    make_room(length + 1);
    for (i = 0; i < length; ++i) {
        unsigned char c = (unsigned char)text[i];
        /* Keep newlines; fold every other control byte to a space so the
         * wrapper and the debug font never see anything unprintable. */
        if (c != '\n' && (c < 32 || c >= 127)) c = ' ';
        transcript[transcript_length++] = (char)c;
    }
    transcript[transcript_length] = '\0';
    wrap_dirty = 1;
}

static void raw_append_str(const char *text) {
    if (text) raw_append(text, strlen(text));
}

static void drop_draft(void) {
    if (transcript_length == committed_length) return;
    transcript_length = committed_length;
    transcript[transcript_length] = '\0';
    wrap_dirty = 1;
}

void falcon_ui_set_draft(const char *text) {
    drop_draft();
    if (committed_length) raw_append_str("\n");
    raw_append_str("You: ");
    raw_append_str(text);
    raw_append_str("_");
}

void falcon_ui_commit_draft(const char *text) {
    drop_draft();
    if (committed_length) raw_append_str("\n");
    raw_append_str("You: ");
    raw_append_str(text);
    raw_append_str("\nAI: ");
    committed_length = transcript_length;
}

void falcon_ui_append(const char *text, size_t length) {
    drop_draft();
    raw_append(text, length);
    committed_length = transcript_length;
}

void falcon_ui_append_str(const char *text) {
    if (text) falcon_ui_append(text, strlen(text));
}

void falcon_ui_end_turn(void) {
    drop_draft();
    raw_append_str("\n");
    committed_length = transcript_length;
}

static void add_line(size_t start, size_t length, int indent) {
    if (line_total >= MAX_WRAP_LINES) return;
    line_start[line_total] = (unsigned short)start;
    line_length[line_total] = (unsigned char)length;
    line_indent[line_total] = (unsigned char)indent;
    ++line_total;
}

static void ensure_wrapped(void) {
    size_t at = 0;
    int continuation = 0;
    if (!wrap_dirty) return;
    line_total = 0;
    while (at < transcript_length) {
        int indent = continuation ? UI_WRAP_INDENT : 0;
        size_t width = (size_t)(UI_COLS - indent);
        size_t remaining = transcript_length - at;
        size_t limit = remaining < width ? remaining : width;
        size_t newline = at, scan;
        int found_newline = 0;
        for (scan = 0; scan < limit; ++scan) {
            if (transcript[at + scan] == '\n') {
                newline = at + scan;
                found_newline = 1;
                break;
            }
        }
        if (found_newline) {
            add_line(at, newline - at, indent);
            at = newline + 1;
            continuation = 0;
            continue;
        }
        if (remaining <= width) {
            add_line(at, remaining, indent);
            at = transcript_length;
            continuation = 0;
            continue;
        }
        /* Break on the last space that fits, otherwise split the word. */
        {
            size_t space = 0;
            int have_space = 0;
            for (scan = 0; scan < width; ++scan) {
                if (transcript[at + scan] == ' ') {
                    space = at + scan;
                    have_space = 1;
                }
            }
            if (have_space && space > at) {
                add_line(at, space - at, indent);
                at = space + 1;
            } else {
                add_line(at, width, indent);
                at += width;
            }
            continuation = 1;
        }
    }
    if (transcript_length && transcript[transcript_length - 1] == '\n')
        add_line(transcript_length, 0, 0);
    wrap_dirty = 0;
}

int falcon_ui_line_count(void) {
    ensure_wrapped();
    return line_total;
}

int falcon_ui_max_scroll(void) {
    int excess;
    ensure_wrapped();
    excess = line_total - UI_CHAT_ROWS;
    return excess > 0 ? excess : 0;
}

static void clear_row(char *row) {
    memset(row, ' ', UI_COLS);
    row[UI_COLS] = '\0';
}

/* Copies text into row at column, sanitizing and clipping to the grid. */
static void put_text(char *row, int column, const char *text, size_t length) {
    size_t i;
    if (column < 0) column = 0;
    for (i = 0; i < length && column + (int)i < UI_COLS; ++i) {
        unsigned char c = (unsigned char)text[i];
        row[column + (int)i] = (c >= 32 && c < 127) ? (char)c : '?';
    }
}

static void put_string(char *row, int column, const char *text) {
    if (text) put_text(row, column, text, strlen(text));
}

static void put_right(char *row, const char *text) {
    int length = text ? (int)strlen(text) : 0;
    if (length > UI_COLS) length = UI_COLS;
    put_text(row, UI_COLS - length, text, (size_t)length);
}

static void fill_rule(char *row, char character) {
    memset(row, character, UI_COLS);
    row[UI_COLS] = '\0';
}

static void compose_header(char *row, const FalconUiState *state) {
    char right[64];
    int filled = 0, i, total = state->context_total, used = state->context_used;
    clear_row(row);
    put_string(row, 0, "LLMPSP  -  Falcon-H1 90M  Q4");
    if (total > 0) {
        filled = used * 10 / total;
        if (filled > 10) filled = 10;
        if (filled < 0) filled = 0;
    }
    {
        char bar[13];
        bar[0] = '[';
        for (i = 0; i < 10; ++i) bar[1 + i] = i < filled ? '#' : '.';
        bar[11] = ']';
        bar[12] = '\0';
        snprintf(right, sizeof(right), "ctx %d/%d %s", used, total, bar);
    }
    put_right(row, right);
}

static void compose_chat_rule(char *row, const FalconUiState *state) {
    char right[64];
    int count = falcon_ui_line_count();
    int first = state->scroll;
    int last = first + UI_CHAT_ROWS;
    fill_rule(row, '-');
    put_string(row, 0, "-- chat ");
    if (count > UI_CHAT_ROWS) {
        if (last > count) last = count;
        snprintf(right, sizeof(right), " %d-%d of %d %s%s ",
                 first + 1, last, count,
                 first > 0 ? "^" : " ",
                 last < count ? "v" : " ");
        put_right(row, right);
    }
}

static void compose_chat(char grid[UI_ROWS][UI_COLS + 1],
                         const FalconUiState *state) {
    int count = falcon_ui_line_count();
    int row;
    for (row = 0; row < UI_CHAT_ROWS; ++row) {
        char *target = grid[UI_ROW_CHAT + row];
        int index = state->scroll + row;
        clear_row(target);
        if (index >= 0 && index < count)
            put_text(target, line_indent[index],
                     transcript + line_start[index], line_length[index]);
    }
}

static void compose_keyboard(char grid[UI_ROWS][UI_COLS + 1],
                             const FalconUiState *state) {
    const char *keys = falcon_ui_keys(state->upper);
    int left = (UI_COLS - UI_KEY_COLUMNS * UI_KEY_CELL) / 2;
    int row, column;
    for (row = 0; row < UI_KEY_ROWS; ++row) {
        char *target = grid[UI_ROW_KEYS + row * UI_KEY_ROW_STEP];
        clear_row(target);
        for (column = 0; column < UI_KEY_COLUMNS; ++column) {
            int index = row * UI_KEY_COLUMNS + column;
            int at = left + column * UI_KEY_CELL;
            char cell[UI_KEY_CELL + 1];
            cell[0] = (!state->busy && index == state->cursor) ? '[' : ' ';
            cell[1] = keys[index];
            cell[2] = (!state->busy && index == state->cursor) ? ']' : ' ';
            cell[3] = ' ';
            cell[4] = '\0';
            put_text(target, at, cell, UI_KEY_CELL);
        }
    }
}

void falcon_ui_compose(const FalconUiState *state,
                       char grid[UI_ROWS][UI_COLS + 1]) {
    int row;
    ensure_wrapped();
    for (row = 0; row < UI_ROWS; ++row) clear_row(grid[row]);

    compose_header(grid[UI_ROW_HEADER], state);
    compose_chat_rule(grid[UI_ROW_SEP_CHAT], state);
    compose_chat(grid, state);

    fill_rule(grid[UI_ROW_SEP_KEYS], '-');
    put_string(grid[UI_ROW_SEP_KEYS], 0,
               state->upper ? "-- keyboard: UPPER " : "-- keyboard: lower ");
    compose_keyboard(grid, state);

    if (state->busy) {
        /* Typing is blocked, so the controls give way to progress. */
        put_string(grid[UI_ROW_HELP], 0, "O  stop the reply");
        put_string(grid[UI_ROW_HELP + 1], 0,
                   state->status ? state->status : "");
    } else {
        put_string(grid[UI_ROW_TOPSTATUS], 0,
                   state->status ? state->status : "");
        put_string(grid[UI_ROW_HELP], 0,
                   "X  add letter     []  space     /\\  backspace     "
                   "L/R  case");
        put_string(grid[UI_ROW_HELP + 1], 0,
                   "START  send     SELECT  new chat     Stick  scroll     "
                   "HOME  quit");
    }
}
