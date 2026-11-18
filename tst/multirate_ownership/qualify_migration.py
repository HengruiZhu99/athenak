#!/usr/bin/env python3
"""Link a lifecycle test main to the production objects in a Makefiles build.

Using the application's objects tests the actual ownership/restart lifecycle
without compiling a second copy of all physics modules or a production test hook.
"""
import argparse
from pathlib import Path
import shlex
import subprocess


def link_main(build, cxx, source, work):
    app_dir = build.resolve() / "src"
    target_dir = app_dir / "CMakeFiles/athena.dir"
    work.mkdir(parents=True, exist_ok=True)
    flags = {}
    for line in (target_dir / "flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            flags[key] = shlex.split(value)
    obj = work / (source.name + ".o")
    binary = work / source.stem
    compile_command = [cxx]
    for group in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        compile_command.extend(flags[group])
    compile_command.extend(["-c", str(source), "-o", str(obj)])
    subprocess.run(compile_command, check=True, cwd=app_dir)
    link = shlex.split((target_dir / "link.txt").read_text())
    original_main = "CMakeFiles/athena.dir/main.cpp.o"
    if link.count(original_main) != 1 or link.count("-o") != 1:
        raise RuntimeError("Unsupported production link command for lifecycle test")
    link[link.index(original_main)] = str(obj)
    link[link.index("-o") + 1] = str(binary)
    subprocess.run(link, check=True, cwd=app_dir)
    return binary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--launcher", default="")
    args = parser.parse_args()
    work = args.build.resolve() / "tst/multirate_ownership/lifecycle"
    binary = link_main(args.build, args.cxx, Path(__file__).with_name("accepted_migration.cpp"), work)
    launch = shlex.split(args.launcher.format(ranks=args.ranks))
    subprocess.run([*launch, str(binary), str(args.fixture.resolve())],
                   check=True, cwd=work, timeout=600)


if __name__ == "__main__":
    main()
