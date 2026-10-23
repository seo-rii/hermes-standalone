# Offline comparison fixtures

These fixtures contain deterministic test inputs and independent reference
outputs for the owned implementations. Provenance JSON in `standalone/`
records source versions, hashes, profiles, licenses and known mismatches.
They do not contain the developer's live service credentials or user state.

Some cryptographic fixtures intentionally contain key-shaped data, JWTs,
certificate/key parser markers and test passwords. In particular,
`azure_certificates.json` encrypts the public TLS test key from the pinned
`tlslite-ng` source archive. Its RSA parameters match that upstream public
test fixture. It must never be used as a real service or user identity.

Recorded model/controller values are comparison inputs for tests. They are
not prerecorded inference outputs used by runtime providers. Full-model,
small-graph, controller-only and individual kernel checks remain distinct in
the provenance and capability reports.
