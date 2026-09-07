LLMPSP - Falcon-H1 Tiny 90M on a PSP (v1.1)

Put this whole folder at:
ms0:/PSP/GAME/LLMPSP/

Required files:
- EBOOT.PBP
- model.fhq4
- llmpsp_config.cfg (optional; defaults are used when missing)

The screen is one page: the conversation fills the top, the keyboard and
controls are at the bottom. What you type shows up straight away as a
"You:" line in the conversation, and the reply is written underneath it.

Controls:
D-pad     move around the keyboard
X         type the highlighted character
Square    space
Triangle  backspace
L / R     switch between lower and upper case
Stick     scroll the conversation up and down
START     send your message
SELECT    start a new conversation (clears the history)
O         stop the reply while it is being written
PS/HOME   quit the app

Replies stream in as they are written and the view follows along. After a
reply finishes, just type again and press START: the conversation
continues, and only your new message is processed, so follow-up replies
start sooner than the first one. The header shows how much of the
conversation space is used; when it runs out, press SELECT.

The status line at the bottom shows progress: "Prompt processing" while
your message is being read, then the live reply rate.

Settings live in llmpsp_config.cfg next to the EBOOT (plain text, one
"key = value" per line, '#' starts a comment; every value is clamped to
a safe range). The interesting ones:
- cache_mb: how many MiB of weights to hold in RAM (0-46, default 44).
  The rest streams from the memory stick. Set 0 if startup crashes.
- context: conversation window in tokens (128-512). Smaller frees RAM
  for more cached weights, which is slightly faster.
- temperature / top_p / top_k / repetition_penalty: 0 temperature gives
  the same reply every time; try temperature = 0.7 for variety.
- max_reply_tokens: longest single reply (8-256).
- cpu_mhz: lower the clock for battery or stability experiments.

Expect roughly half a word per second on a PSP-2000/3000 at default
settings; the first reply is slower because the whole prompt is
processed first.

Diagnostics:
- The app appends progress to llmpsp_trace.txt next to EBOOT.PBP. If the
  PSP ever powers off, the last line of that file shows exactly how far
  it got, and battery/power events are recorded there too. Each turn
  also logs a [perf] timing breakdown.
- If model.fhq4 is missing or truncated the app says so on startup
  instead of failing later.

Model weights are covered by the Falcon LLM License, not the code's MIT
license. See MODEL_NOTICE.txt.
