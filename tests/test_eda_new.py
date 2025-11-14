from src.attacks.eda_new import UserContentQuery


def test_user_content_query_validator():
    # Test removing trusted bids from untrusted
    query = UserContentQuery(
        has_trusted_user_content=True,
        has_untrusted_user_content=True,
        trusted_user_content_bids=[1, 2, 3],
        untrusted_user_content_bids=[2, 3, 4, 5],
    )
    assert query.untrusted_user_content_bids == [4, 5], "Should remove trusted bids from untrusted"
    assert query.trusted_user_content_bids == [1, 2, 3], "Should not modify trusted bids"

    # Test empty lists when boolean flags are False
    query = UserContentQuery(
        has_trusted_user_content=False,
        has_untrusted_user_content=False,
        trusted_user_content_bids=[1, 2, 3],
        untrusted_user_content_bids=[4, 5, 6],
    )
    assert query.trusted_user_content_bids == [], "Should empty trusted bids when has_trusted_user_content is False"
    assert query.untrusted_user_content_bids == [], (
        "Should empty untrusted bids when has_untrusted_user_content is False"
    )

    # Test mixed case - only trusted content
    query = UserContentQuery(
        has_trusted_user_content=True,
        has_untrusted_user_content=False,
        trusted_user_content_bids=[1, 2, 3],
        untrusted_user_content_bids=[2, 4, 5],
    )
    assert query.trusted_user_content_bids == [1, 2, 3], "Should keep trusted bids"
    assert query.untrusted_user_content_bids == [], "Should empty untrusted bids due to flag"
