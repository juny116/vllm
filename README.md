# vLLM — Janus-Pro and text-only model support

[vLLM](https://github.com/vllm-project/vllm)을 기반으로 Janus-Pro 모델 연결과 멀티모달 모델의 텍스트 전용 입력 처리를 추가한 추론 서버 코드입니다.

## 추가·수정한 기능

- `vllm/model_executor/models/janus_pro.py`: Janus-Pro 모델 구현.
- `vllm/model_executor/models/registry.py`: 모델 레지스트리에 Janus-Pro 등록.
- `vllm/transformers_utils/configs/janus_pro.py`: 모델 설정 클래스.
- `vllm/transformers_utils/config.py`, `configs/__init__.py`: 설정 클래스 연결.
- `vllm/transformers_utils/chat_templates/template_janus_pro.jinja`: Janus-Pro 대화 템플릿.
- `vllm/transformers_utils/chat_templates/registry.py`: 대화 템플릿 등록.
- `vllm/model_executor/models/transformers/multimodal.py`: 이미지 없는 입력의 처리 분기 추가.

## 코드 구조

| 경로 | 역할 |
|---|---|
| `vllm/entrypoints/` | CLI와 API 서버 진입점 |
| `vllm/model_executor/` | 모델 로딩과 실행 |
| `vllm/transformers_utils/` | 모델 설정·토크나이저·대화 템플릿 |
| `vllm/engine/`, `vllm/v1/` | 추론 요청 처리와 실행 엔진 |
| `csrc/` | 네이티브 연산 구현 |
| `tests/` | 모델·서버·연산 테스트 |
| `examples/` | API 및 오프라인 추론 예시 |

## 사용

이 저장소의 변경을 사용하려면 이 소스 트리에서 vLLM을 빌드해야 합니다. 하드웨어별 빌드 절차와 실행 예시는 [기반 프로젝트 안내](README_UPSTREAM.md)와 [docs](docs/)를 참고합니다.

```bash
vllm serve <model-name-or-path>
```

라이선스는 [LICENSE](LICENSE)에 있습니다.
