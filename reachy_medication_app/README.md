---
title: Reachy Medication App
emoji: 🤖
colorFrom: purple
colorTo: gray
sdk: static
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Reachy Medication App

This is the working app for my medication project. I forked it from the Reachy Mini
conversation app and added a medication workflow on top of it: scanning bottle
labels, verifying doses, counting pills, setting reminders, and checking dose
history. The robot only verifies and reminds. It does not give medical advice.

All of my custom work lives in
`src/reachy_medication_app/profiles/_reachy_medication_app_locked_profile`:
- `instructions.txt` is the system prompt with the workflow and the safety rules.
- `tools.txt` lists the tools the robot can use.
- The medication tools (`scan_bottle`, `add_medication`, `verify_dose`,
  `count_pills`, `set_reminder`, `check_dose_history`) are the `.py` files in that
  same folder. Each one subclasses the `Tool` class.

The Gemini wrapper and the in-memory medication store are in
`src/reachy_medication_app/medication/`.

To run it you need a `.env` file with your own API keys. See `.env.example` for the
format. My real `.env` is not committed.

The original README from the conversation app I forked is kept in
`../archive/app_template_leftovers/README_OLD.md`.