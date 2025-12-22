from typing import Any, Dict

from .base import Skill


class LoginFormSkill(Skill):
    name = "login_form_skill"

    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        page = context["page"]
        username = context["username"]
        password = context["password"]
        target_url = context.get("url")

        if target_url:
            try:
                current_url = page.url
            except Exception:
                current_url = None
            if not current_url or target_url not in current_url:
                await page.goto(target_url, wait_until="domcontentloaded", timeout=12000)

        await page.wait_for_selector("input#username", timeout=6000)
        await page.fill("input#username", username)
        await page.wait_for_selector("input#password", timeout=6000)
        await page.fill("input#password", password)
        await page.wait_for_selector("button#submit", timeout=6000)
        await page.click("button#submit")

        return {
            "status": "success",
            "skill": self.name,
        }
