import os
import sys
import fnmatch

def print_directory_tree(root_dir, excluded_dirs=None, excluded_files=None, prefix=''):
    if excluded_dirs is None:
        excluded_dirs = [
            'node_modules', '__pycache__', '.git', '.hg', '.svn',
            '.egg-info', '.eggs', '.cache', '.yarn', 'tests',
            'venv', '.venv', 'env', '.env', 'virtualenv',
            'Lib', 'lib', 'Include', 'Scripts', 'bin', 'share', 'lib64', 'site-packages',
            'build', 'dist', 'logs', '.benchmarks', '.coverage', '.pytest_cache', '.vscode',
            '.idea', 'Dockerfile*', 'Makefile', 'Procfile', '*.egg-info'
        ]
    if excluded_files is None:
        excluded_files = [
            'package-lock.json', 'yarn.lock', '*.pyc', '*.log',
            '*.gz', '*.zip', '*.png', '*.jpg', '*.jpeg', '*.gif',
            '*.md', '*.rst', 'LICENSE', '*.txt', 'README*', '*.ini', '*.cfg', '*.yml', '*.yaml', '*.db', '*.sqlite'
        ]
    basename = os.path.basename(os.path.normpath(root_dir))
    print(prefix + basename + '/')
    prefix_cont = prefix + '    '
    try:
        entries = os.listdir(root_dir)
    except PermissionError:
        entries = []
    entries = sorted(entries)
    # Filter out excluded directories
    entries = [
        e for e in entries
        if not any(
            fnmatch.fnmatch(e, pattern) or e.lower() == pattern.lower()
            for pattern in excluded_dirs
        )
    ]
    for index, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        # Skip excluded files based on patterns
        if os.path.isfile(path):
            if any_matches(entry, excluded_files):
                continue
        elif os.path.isdir(path):
            if any(
                fnmatch.fnmatch(entry, pattern) or entry.lower() == pattern.lower()
                for pattern in excluded_dirs
            ):
                continue
        connector = '|-- ' if index < len(entries) - 1 else '`-- '
        if os.path.isdir(path):
            print(prefix_cont + connector + entry + '/')
            print_directory_tree(
                path, excluded_dirs, excluded_files, prefix_cont + ('|   ' if index < len(entries) - 1 else '    ')
            )
        else:
            print(prefix_cont + connector + entry)

def any_matches(filename, patterns):
    return any(fnmatch.fnmatch(filename, pattern) or filename.lower() == pattern.lower() for pattern in patterns)

if __name__ == '__main__':
    # Set the root directory to the first command-line argument or the current directory
    if len(sys.argv) > 1:
        root_directory = sys.argv[1]
    else:
        root_directory = '.'
    print_directory_tree(root_directory)
