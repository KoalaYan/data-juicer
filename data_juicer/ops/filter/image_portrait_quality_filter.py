from typing import List, Optional, Union

from data_juicer.utils.constant import Fields, MetaKeys

from ..base_op import NON_STATS_FILTERS, OPERATORS, Filter

OP_NAME = "image_portrait_quality_filter"


@NON_STATS_FILTERS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class ImagePortraitQualityFilter(Filter):
    """Keep samples based on upstream portrait hard-quality triage.

    Run ``image_portrait_quality_mapper`` first. By default this filter keeps
    both ``pass`` and ``uncertain`` samples and removes only high-confidence
    ``reject`` samples. ``any_or_all`` controls samples containing multiple
    images.
    """

    def __init__(
        self,
        quality_key: str = MetaKeys.portrait_quality,
        keep_statuses: Union[str, List[str]] = ("pass", "uncertain"),
        keep_human_statuses: Optional[Union[str, List[str]]] = None,
        any_or_all: str = "all",
        keep_missing: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(keep_statuses, str):
            keep_statuses = [keep_statuses]
        if isinstance(keep_human_statuses, str):
            keep_human_statuses = [keep_human_statuses]
        self.quality_key = quality_key
        self.keep_statuses = set(keep_statuses)
        self.keep_human_statuses = set(keep_human_statuses) if keep_human_statuses else None
        self.keep_missing = keep_missing
        if any_or_all not in {"any", "all"}:
            raise ValueError("any_or_all must be one of ['any', 'all']")
        self.any = any_or_all == "any"

    def compute_stats_single(self, sample):
        return sample

    def process_single(self, sample):
        quality_records = (sample.get(Fields.meta) or {}).get(self.quality_key)
        if not quality_records:
            return self.keep_missing
        decisions = [
            record.get("status") in self.keep_statuses
            and (
                self.keep_human_statuses is None
                or record.get("human_status") in self.keep_human_statuses
            )
            for record in quality_records
        ]
        return any(decisions) if self.any else all(decisions)
