
## Version 1.0.3 -- 2025-07-02
*   Fixed bug in `hetGP.predict` for kriging variance (`sd2`) under simple kriging trendtype (thanks to Ozge Surer, Miami Ohio, for flagging)

## Version 1.0.2 -- 2025-01-14
*   Fixed bug in `src/matern.cpp` for partial derivatives

## Version 1.0.1 -- 2025-01-07

*   Added `ci.yaml` for continuous integration/wheelbuilding for distributing binaries
*   Reworked workflow files to run basic and advanced tests under different conditions

## Version 1.0.0 -- 2024-12-18

*   Initial release of `hetGPy`
*   Many bug fixes, improvements, and added documentation
*   Special thanks to @eidenhofer and @DanWaxman for their careful and thoughtful review