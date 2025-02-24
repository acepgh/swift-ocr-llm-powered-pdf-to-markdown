{pkgs}: {
  deps = [
    pkgs.geckodriver
    pkgs.tesseract
    pkgs.poppler_utils
    pkgs.libGLU
    pkgs.libGL
    pkgs.libxcrypt
  ];
}
