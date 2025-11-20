본 프로젝트는 [DevContainer](https://containers.dev/), [Git](https://git-scm.com/), [uv](https://docs.astral.sh/uv/), [Ruff](https://docs.astral.sh/ruff/)를 사용해 프로젝트를 관리합니다. 기본적인 작업 흐름은 다음과 같습니다.

1. 추가할 기능에 대응되는 브랜치를 생성한다: `git checkout -b <name>/<feat>`
2. 브랜치에 변경 사항을 커밋한다: `git add`, `git commit`
3. [Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) (PR)를 작성하고, 코드 리뷰를 받는다.
4. PR을 반영하고, 브랜치를 삭제한다.

각 툴과 작업 흐름에 대한 자세한 설명은 개별 문서를 참고 바랍니다.
- [Github_Flow](docs/github_flow.md)
- [uv](docs/uv.md)
- [Ruff](docs/static_checker.md)
