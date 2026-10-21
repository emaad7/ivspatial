import sys

if len(sys.argv) == 1:
    from .gui import main
    main()
else:
    from .cli import main
    sys.exit(main())
