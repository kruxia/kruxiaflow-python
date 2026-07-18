# Changelog

All notable changes to the Kruxia Flow Python SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - TBD

First PyPI release.

### Changed

- **Distribution renamed** from `kruxiaflow-python` (GitHub-install only) to
  `kruxiaflow` on PyPI. The import package is unchanged (`import kruxiaflow`);
  existing code needs no modification.
- **License changed from MIT to Apache-2.0**, unifying with the Kruxia Flow
  engine license. Versions prior to 0.4.0 remain MIT. Added a `NOTICE` file.

### Added

- Release automation: the `kruxia/kruxiaflow-py-std` worker image is now built and
  pushed automatically on each release, version-tagged to match the SDK
  (amd64; arm64 deferred).
- `CHANGELOG.md` (this file).

## [0.3.0] - 2026

Final GitHub-install release under the MIT license (`kruxiaflow-python`).
Workflow definition DSL, `ScriptActivity`, and the worker SDK
(poll/heartbeat/complete loop with `@activity` registration).
