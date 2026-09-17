import sys


def _launch() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "app":
        from vsl_study.desktop import main as desktop_main

        return desktop_main()
    from vsl_study.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_launch())
