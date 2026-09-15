# Axial-layout study evidence

Lossless evidence for generation `b23aca5d-3315-42c4-832f-8668c8b36ceb`.
The report and code are on [main](https://github.com/Shakil-Muhit/axial-layout-scoping/blob/main/RESULT.md).

Run `sha256sum -c SHA256SUMS`, join `artifacts.tar.gz.part*` in lexical order,
and extract the resulting gzip tar archive into the main checkout's `evidence/`.
Full restoration commands are in `main:evidence/README.md`.

Archive SHA256: `478cc56b5ee5ff4fcc962f92db0cc33fdc07d2765702848e952ee799b4dd054c`.
`ARTIFACTS.json` contains hashes for every part and every original file.

This branch contains generated artifacts; the main branch retains its
registered evidence ignore policy. Contents include the rejected diagnostic
trace as well as the unchanged baseline, so that exclusions are auditable.
