PYTHON ?= python3

.PHONY: list doctor plan-smoke dry-run-smoke

list:
	$(PYTHON) benchctl.py list --set daily-full

doctor:
	$(PYTHON) benchctl.py doctor --set smoke

plan-smoke:
	$(PYTHON) benchctl.py plan --set smoke --backend triton_experimental --devices 7 --run-id smoke-plan

dry-run-smoke:
	$(PYTHON) benchctl.py run --set smoke --backend triton_experimental --devices 7 --run-id smoke-dry-$$(date +%s) --dry-run
