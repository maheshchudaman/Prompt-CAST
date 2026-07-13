import argparse

from castnet.reproducibility import environment_record, git_revision, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    record = environment_record()
    record["git_commit"] = git_revision()
    write_json(args.output, record)


if __name__ == "__main__":
    main()
