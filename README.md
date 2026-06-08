# Reachy Mini Medication Project

**Demo video:** https://youtu.be/kM0BgfjUoeA

This is my independent study project for my M.S. in Computer Science. I wanted to
see if a small tabletop robot (the Reachy Mini) with a vision-language model could
help older adults *check* their medications at home, not just remind them. So the
robot reads the bottle label, counts the pills, makes sure the dose looks right, and
sets reminders. It does not try to act like a doctor.

I started with a lit review on how older adults manage medication and what tools
they already use. That is where I found the gap: most of the technology out there
reminds people to take a dose, but almost nothing actually uses vision to verify that
the pills in front of them are the right ones. That became the focus of the project.

From there I ran two experiments. The first compared ChatGPT, Claude, and Gemini on
counting and identifying pills, and Gemini came out clearly on top. The second tested
whether different prompts changed Gemini's accuracy, and they mostly did not. When I
started building the actual app with Reachy, the counts were sometimes wrong even
though Gemini had been accurate on my phone. So I ran a comparison test to figure out
whether the problem was Reachy's camera, the way the image was sent to the model, or
the pills themselves. It turned out the only real failure case is when the pills
overlap. After that I built out the full medication app and recorded three demos.

## What's in each folder

| Folder | What's in it |
|---|---|
| `docs/` | My literature review (25 papers) |
| `experiments/` | `pill_counting_experiments.xlsx`, the cleaned-up version of all my experiment data. The original exports are in `source_exports/` |
| `figures/` | Images for the paper: the phone-vs-Reachy comparison shots and a frame from Reachy's camera |
| `paper/` | My final paper (LaTeX source plus the compiled PDF) |
| `scripts/` | Two earlier standalone prototypes I wrote before the full app |
| `reachy_medication_app/` | The actual working app. A voice-conversational medication assistant built on top of the Reachy conversation framework |
| `archive/` | Older notes and raw prototype output. Not needed to run anything, just kept for reference |

A note on this folder: it is also a Python virtual environment, so `bin/`, `lib/`,
and `pyvenv.cfg` are environment files. Leave those alone.

## The project, step by step

1. **Lit review.** Found the gap: lots of tech reminds people, almost none verifies
   the right pills with vision on a real home robot.
2. **Experiment 1, model comparison.** ChatGPT vs Claude vs Gemini, 15 images and 3
   prompts. Gemini was the most accurate. Overlapping and touching pills were the
   hardest.
3. **Experiment 2, prompting study.** Gemini with 10 different prompts. Prompting on
   its own did not reliably fix the counting errors.
4. **Reachy camera comparison test.** Tested where the counts actually break. It was
   only the dense, overlapping layouts, not the camera or the code.
5. **Built the app.** Bottle scan, dose verification, reminders, and safety rules.
6. **Three demos.** (1) read a label, save the medication, set a reminder. (2) verify
   a dose, log it, and tell whether it was already taken. (3) catch a wrong pill
   count, explain it, then confirm the count after I fix it.

## Running the app

The app needs a Reachy Mini (or its simulator) and your own API keys. Copy
`reachy_medication_app/.env.example` to `reachy_medication_app/.env` and fill in your
Gemini and OpenAI keys, then install and run the app from inside
`reachy_medication_app/` (it uses `pyproject.toml`). My real `.env` with keys is not
included here on purpose.
