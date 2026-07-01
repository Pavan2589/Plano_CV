import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger("plano_cv.alignment")
logger.setLevel(logging.INFO)


class ImageAligner:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def align(self, reference_image: Image.Image, check_image: Image.Image):
        """
        Computes a homography that warps check_image into reference_image's
        coordinate space, so expected_item bboxes (defined in reference
        image coordinates) can be directly applied to crop the warped
        check_image at the same locations.

        Returns: (warped_check_image_as_PIL, success: bool, match_count: int)
        If alignment fails (too few good matches), returns the original
        check_image unmodified with success=False, so callers can fall
        back gracefully instead of crashing.
        """
        ref_gray = cv2.cvtColor(np.array(reference_image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        chk_gray = cv2.cvtColor(np.array(check_image.convert("RGB")), cv2.COLOR_RGB2GRAY)

        kp1, des1 = self.orb.detectAndCompute(ref_gray, None)
        kp2, des2 = self.orb.detectAndCompute(chk_gray, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            logger.warning("Alignment failed: insufficient keypoints detected")
            return check_image, False, 0

        matches = self.matcher.knnMatch(des1, des2, k=2)

        good_matches = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        MIN_MATCH_COUNT = 15
        if len(good_matches) < MIN_MATCH_COUNT:
            logger.warning(f"Alignment failed: only {len(good_matches)} good matches "
                             f"(need {MIN_MATCH_COUNT})")
            return check_image, False, len(good_matches)

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            logger.warning("Alignment failed: homography computation returned None")
            return check_image, False, len(good_matches)

        ref_width, ref_height = reference_image.size
        chk_array = np.array(check_image.convert("RGB"))
        warped = cv2.warpPerspective(chk_array, H, (ref_width, ref_height))

        inlier_count = int(mask.sum()) if mask is not None else len(good_matches)
        logger.info(f"Alignment succeeded: {inlier_count} inlier matches out of "
                     f"{len(good_matches)} good matches")

        return Image.fromarray(warped), True, inlier_count


aligner_instance = ImageAligner()
