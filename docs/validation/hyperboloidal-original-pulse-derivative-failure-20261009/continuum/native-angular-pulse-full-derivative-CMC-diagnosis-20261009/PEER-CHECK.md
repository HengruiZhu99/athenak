The literature agent independently checked the following source/pencil point on 2026-10-09; no import, arithmetic run, root solve or scientific payload read was performed by that reviewer.

`ControlGraph.source` uses scalar `q=vc.norm(y)` in `b=omega*q/a` while H, omega and nu are Jet2 functions of Y. The factored identity is valid only when `b=omega*sqrt(sum(Y_i^2))/a` as a jet in the q>0 branch. Keeping q scalar omits radial first/second derivatives. `direct_D` retains full smooth bnu jets. The proposed q>0-only jet-radius replacement is the minimal algebraic correction; the exact-center direct branch remains valid.

This peer check supports the source diagnosis only; it does not qualify the failed full gate or approve execution of a repaired control gate.
