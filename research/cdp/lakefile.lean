import Lake

open Lake DSL

package «cdp-paper-proofs» where
  leanOptions := #[⟨`autoImplicit, false⟩]

require "leanprover-community" / "mathlib" @ git "v4.29.1"

@[default_target]
lean_lib «GateSecurity» where srcDir := "proofs/security"
@[default_target]
lean_lib «SerialGateSecurity» where srcDir := "proofs/security"
@[default_target]
lean_lib «ModelSecurity» where srcDir := "proofs/security"
@[default_target]
lean_lib «ModelSecurityV2» where srcDir := "proofs/security"
@[default_target]
lean_lib «ModelSecurityV3» where srcDir := "proofs/security"
@[default_target]
lean_lib «ModelSecurityV4» where srcDir := "proofs/security"
@[default_target]
lean_lib «DistributedSecurity» where srcDir := "proofs/security"
@[default_target]
lean_lib «DistributableClaims» where srcDir := "proofs/security"
@[default_target]
lean_lib «CapacitySecurity» where srcDir := "proofs/security"
@[default_target]
lean_lib «BiometricSecurity» where srcDir := "proofs/security"
@[default_target]
lean_lib «CompressionSecurity» where srcDir := "proofs/security"
@[default_target]
lean_lib «GatePlacement» where srcDir := "proofs/security"
@[default_target]
lean_lib «GenerationIsolation» where srcDir := "proofs/security"
@[default_target]
lean_lib «MatrixIsolation» where srcDir := "proofs/security"
@[default_target]
lean_lib «DictionaryRegime» where srcDir := "proofs/security"
@[default_target]
lean_lib «AttentionLeakage» where srcDir := "proofs/security"
@[default_target]
lean_lib «MaskBoolean» where srcDir := "proofs/security"
@[default_target]
lean_lib «MultiEncoder» where srcDir := "proofs/security"

@[default_target]
lean_lib «EnglishTax» where srcDir := "proofs/channels"
@[default_target]
lean_lib «SpileLattice» where srcDir := "proofs/channels"
@[default_target]
lean_lib «RegimeGrowth» where srcDir := "proofs/channels"
