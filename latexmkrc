$out_dir = 'build';
$out2_dir = '.';

# Keep LaTeX runs non-interactive and show exact file:line errors.
$xelatex = 'xelatex -interaction=nonstopmode -file-line-error %O %S';

# Ensure output directory exists before running LaTeX/Biber.
ensure_path($out_dir);
