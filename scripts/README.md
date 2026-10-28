# Shared scripts

Place utilities used across encoders here when needed. Encoder-specific scripts
live under `encoders/<encoder>/scripts/`; the current Qwen scripts are in
`encoders/Qwen3.8-vision-compression/scripts/`.

Run scripts from the repository root so relative input and output paths resolve
consistently. Do not add a shared framework before experiments need one.
