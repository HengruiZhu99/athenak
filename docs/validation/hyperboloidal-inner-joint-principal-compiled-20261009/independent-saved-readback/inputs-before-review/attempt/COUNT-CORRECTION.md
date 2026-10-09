# Source-preparation count correction

The first held proposal is preserved byte for byte. Its ordinary fixed grid has
3 alpha values x2 chi values x3 W values x2 G0 values x2 frames =72 cases.
Adding8 mu1 cases,2 q=f cases and36 general cases yields118, rather than262.
The original source-only proposal therefore failed its own count prerequisite;
no gate import, compile, exact-algebra run or kernel query had occurred.

The exact count-only diff changes the final C++ count assertion and two prose
counts. The probe points, source formulas, exact checker and thresholds remain
unchanged. The new runner/analyzer/recipe are separately prepared for root
review and are not an authorization to execute.
