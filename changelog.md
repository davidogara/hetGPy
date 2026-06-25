
## Verion 1.0.5rc1 2026-06-25
*   Fixed bug in hetGP.predict to properly return `sd2var` when `noise_var=True`
*   Removed `iprint` in `scipy.optimize.minimize`
*   Confirmed that numpy `matmul` warnings are fixed for `numpy>=2.3.0` (requires python>=3.11)

## Version 1.0.5 2026-05-31
*	Fixed bug in `hetGP.mle` (pull request 32, thanks to Marie Cloet, KU Leuven for flagging and for fixing)
	
## Version 1.0.4 -- 2025-07-02
*   Fixed bug in `hetGP.predict` for kriging variance (`sd2`) under simple kriging trendtype (thanks to Ozge Surer, Miami Ohio, for flagging)


## Version 1.0.3 -- 2025-01-31
*   Clarified verbosity of print statements for optimization routines (thanks to Dan Waxman)

## Version 1.0.2 -- 2025-01-14
*   Fixed bug in `src/matern.cpp` for partial derivatives

## Version 1.0.1 -- 2025-01-07

*   Added `ci.yaml` for continuous integration/wheelbuilding for distributing binaries
*   Reworked workflow files to run basic and advanced tests under different conditions

## Version 1.0.0 -- 2024-12-18

*   Initial release of `hetGPy`
*   Many bug fixes, improvements, and added documentation
*   Special thanks to @eidenhofer and @DanWaxman for their careful and thoughtful review
