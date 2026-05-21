"""
print_structure.py

Prints a detailed folder/file tree for any directory, in the same
style as the markdown tree format. Run from anywhere.

Usage:
    python print_structure.py                  # prints from current directory
    python print_structure.py path/to/folder   # prints from that folder
    python print_structure.py --depth 3        # limit depth (default: unlimited)
    python print_structure.py --dirs-only      # skip files, show folders only
    python print_structure.py --save           # also save output to structure.md

Examples:
    python print_structure.py C:/Users/ekowd/Desktop/FYP/FYP
    python print_structure.py . --depth 4
    python print_structure.py . --save
"""

import os
import sys
import argparse
from datetime import datetime


# Folders/files to always skip
IGNORE = {
    '__pycache__', '.git', '.idea', '.vscode',
    'node_modules', '.ipynb_checkpoints',
    'myenv', 'venv', '.env', 'env',
}

# File extensions to mark with a note
NOTABLE_EXTENSIONS = {
    '.py':      '',
    '.ipynb':   '',
    '.keras':   '  [model weights]',
    '.h5':      '  [model weights]',
    '.json':    '',
    '.csv':     '',
    '.txt':     '',
    '.md':      '',
    '.zip':     '',
    '.png':     '',
    '.jpg':     '',
    '.jpeg':    '',
}


def build_tree(root, prefix='', depth=None, current_depth=0,
               dirs_only=False, count=None):
    """
    Recursively builds the tree lines for a directory.

    Returns list of strings (one per line).
    """
    if count is None:
        count = {'files': 0, 'dirs': 0}

    if depth is not None and current_depth >= depth:
        return [], count

    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return [prefix + '    [permission denied]'], count

    # Split into dirs and files
    dirs  = [e for e in entries if os.path.isdir(os.path.join(root, e))
             and e not in IGNORE]
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))
             and e not in IGNORE]

    lines = []

    # Process directories first
    for i, d in enumerate(dirs):
        is_last_dir  = (i == len(dirs) - 1) and (dirs_only or len(files) == 0)
        connector    = '└── ' if is_last_dir else '├── '
        child_prefix = prefix + ('    ' if is_last_dir else '│   ')

        lines.append(f"{prefix}{connector}{d}/")
        count['dirs'] += 1

        subtree, count = build_tree(
            os.path.join(root, d),
            prefix=child_prefix,
            depth=depth,
            current_depth=current_depth + 1,
            dirs_only=dirs_only,
            count=count,
        )
        lines.extend(subtree)

    # Process files (unless dirs_only)
    if not dirs_only:
        for i, f in enumerate(files):
            is_last     = (i == len(files) - 1)
            connector   = '└── ' if is_last else '├── '
            ext         = os.path.splitext(f)[1].lower()
            note        = NOTABLE_EXTENSIONS.get(ext, '')
            lines.append(f"{prefix}{connector}{f}{note}")
            count['files'] += 1

    return lines, count


def print_structure(root, depth=None, dirs_only=False, save=False):
    root = os.path.abspath(root)
    root_name = os.path.basename(root) or root

    header = f"{root_name}/"
    lines, count = build_tree(
        root, depth=depth, dirs_only=dirs_only
    )

    output = [header] + lines
    output_str = '\n'.join(output)

    # Summary footer
    footer = (
        f"\n{'─'*50}\n"
        f"  {count['dirs']} folder(s)  |  {count['files']} file(s)\n"
        f"  Root: {root}\n"
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    if depth:
        footer += f"  Depth limit: {depth}\n"
    if dirs_only:
        footer += f"  Mode: directories only\n"

    print(output_str)
    print(footer)

    if save:
        out_path = os.path.join(os.getcwd(), 'structure.md')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('```\n')
            f.write(output_str)
            f.write('\n```\n')
            f.write(footer)
        print(f"  Saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Print a detailed folder/file tree for any directory.'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Root directory to print. Defaults to current directory.'
    )
    parser.add_argument(
        '--depth', '-d',
        type=int,
        default=None,
        help='Maximum depth to traverse. Default: unlimited.'
    )
    parser.add_argument(
        '--dirs-only',
        action='store_true',
        help='Show directories only, skip files.'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save output to structure.md in the current directory.'
    )

    args = parser.parse_args()
    print_structure(
        root=args.path,
        depth=args.depth,
        dirs_only=args.dirs_only,
        save=args.save,
    )


if __name__ == '__main__':
    main()
