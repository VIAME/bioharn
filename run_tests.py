#!/usr/bin/env python
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    import pytest
    import sys
    package_name = 'bioharn'
    pytest_args = [
        '-p', 'no:doctest',
        '--cov-config', '.coveragerc',
        '--cov-report', 'html',
        '--cov-report', 'term',
        '--xdoctest',
        '--cov=' + package_name,
    ]
    pytest_args = pytest_args + sys.argv[1:]
    sys.exit(pytest.main(pytest_args))
