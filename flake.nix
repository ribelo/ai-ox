{
  description = "A flake for gemini-ox";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          system = "x86_64-linux";
          pkgs = import nixpkgs {
            inherit system;
          };
          native-libs = with pkgs; [ cmake pkg-config ];

          libs = with pkgs; [
            uv
            alsa-lib.dev
            portaudio
            (python3.withPackages (ps: with ps; [
            ]))
          ];
        in
        with pkgs;
        {
          devShells.default = mkShell {
            nativeBuildInputs = native-libs;
            buildInputs = libs;
            shellHook = ''
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath libs}:./"
              export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib.outPath}/lib:$LD_LIBRARY_PATH"
            '';
          };
        });
}
