import json

import pytest

from eikon_engine.skills.extraction.listing_extraction import ListingExtractionSkill

LISTING_HTML = """
<html>
  <body>
    <div class="company-card card">
      <h2>Atlas Robotics</h2>
      <p>Autonomous warehouse robots.</p>
      <a href="https://atlas.ai">Visit</a>
      <p>Founders: Jane Doe and John Roe</p>
    </div>
    <div class="company-card card">
      <h2>Zephyr Labs</h2>
      <p>AI generated food insights.</p>
      <a href="https://zephyr.example">Visit</a>
    </div>
  </body>
</html>
"""

YC_DIRECTORY_HTML = """
<div class="_results_i9oky_343">
  <a class="_company_i9oky_355" href="/companies/airbnb">
    <div class="lg:max-w-[90%]">
      <span class="_coName_i9oky_470">Airbnb</span>
      <p class="_coTagline_i9oky_512">Marketplace for short-term stays.</p>
    </div>
  </a>
  <a class="_company_i9oky_355" href="/companies/doordash">
    <div class="lg:max-w-[90%]">
      <span class="_coName_i9oky_470">DoorDash</span>
      <p class="_coTagline_i9oky_512">Food delivery from local restaurants.</p>
    </div>
  </a>
</div>
"""


@pytest.mark.asyncio
async def test_listing_extraction_skill_persists_selected_card(tmp_path) -> None:
    skill = ListingExtractionSkill()
    artifact_path = tmp_path / "listing.json"
    context = {
        "html": LISTING_HTML,
        "artifact_path": str(artifact_path),
        "page_url": "https://www.ycombinator.com/companies",
    }

    result = await skill.execute(context)

    assert result["status"] == "success"
    assert result["result"]["company_name"] == "Atlas Robotics"
    assert result["result"]["name"] == "Atlas Robotics"
    assert result["result"]["founders"] == ["Jane Doe", "John Roe"]
    saved = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert saved["source_url"] == "https://atlas.ai"
    assert saved["company_name"] == "Atlas Robotics"
    assert saved["founders"] == ["Jane Doe", "John Roe"]


@pytest.mark.asyncio
async def test_listing_extraction_skill_handles_anchor_cards() -> None:
    skill = ListingExtractionSkill()

    result = await skill.execute({"html": YC_DIRECTORY_HTML})

    assert result["status"] == "success"
    assert result["items_found"] == 2
    assert result["result"]["company_name"] == "Airbnb"
    assert result["result"]["source_url"] == "https://www.ycombinator.com/companies/airbnb"
    assert "Marketplace for short-term stays" in result["result"]["description"]
    assert result["result"]["founders"] == []


@pytest.mark.asyncio
async def test_listing_extraction_skill_falls_back_to_page_url_when_missing_link() -> None:
    html = """
    <div class="company-card card">
      <h2>Nova Labs</h2>
      <p>AI for lab automation.</p>
    </div>
    """
    skill = ListingExtractionSkill()

    result = await skill.execute({"html": html, "page_url": "https://www.ycombinator.com/companies"})

    assert result["status"] == "success"
    assert result["result"]["source_url"] == "https://www.ycombinator.com/companies"
    assert result["result"]["company_name"] == "Nova Labs"
    assert result["result"]["founders"] == []
