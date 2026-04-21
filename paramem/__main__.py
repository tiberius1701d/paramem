"""Makes ``python -m paramem`` equivalent to the ``paramem`` console script."""

from paramem.cli.main import main

raise SystemExit(main())
