# Validate canonical, headerless sacct -P rows for one exact allocation.
BEGIN { FS = "|"; max_step = -1 }
function terminal(state, normalized) {
  normalized = state; sub(/\+.*/, "", normalized)
  return normalized ~ /^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)$/
}
{
  if (NF != 12) malformed++
  if (($1 ":") == (job ":")) {
    top++; top_terminal += terminal($5); top_name += ($2 == expected_name)
    top_completed += ($5 == "COMPLETED" && $6 == "0:0")
    top_failed += ($5 != "COMPLETED" && $6 != "0:0")
  } else if (($1 ":") == (job ".extern:")) {
    external++; external_terminal += terminal($5)
  } else if ($1 ~ ("^" job "\\.[0-9]+$")) {
    step_index = $1; sub("^" job "\\.", "", step_index); step_index += 0
    if (seen[step_index]++) duplicates++
    steps++; steps_terminal += terminal($5)
    steps_completed += ($5 == "COMPLETED" && $6 == "0:0")
    if (step_index > max_step) max_step = step_index
  } else { unexpected++ }
}
END {
  common = (malformed == 0 && unexpected == 0 && duplicates == 0 &&
            top == 1 && top_terminal == 1 && top_name == 1 &&
            external == 1 && external_terminal == 1 && steps <= expected)
  contiguous = (steps == 0)
  if (steps > 0 && max_step == steps - 1) {
    contiguous = 1
    for (step_index = 0; step_index < steps; ++step_index)
      if (!(step_index in seen)) contiguous = 0
  }
  if (mode == "success")
    valid = (common && top_completed == 1 && steps == expected &&
             steps_completed == expected)
  else if (mode == "failure")
    valid = (common && top_failed == 1 && contiguous && steps_terminal == steps)
  else valid = 0
  if (!valid) exit 1
  printf("mode=%s numbered_steps=%d\n", mode, steps)
}
