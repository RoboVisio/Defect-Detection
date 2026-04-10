#include "HalconCpp.h"

using namespace HalconCpp;

void Preprocess_Gray_WithROI(HObject ho_Image,
                             HObject* ho_GrayROI,
                             HTuple* hv_ROI_Row1,
                             HTuple* hv_ROI_Col1,
                             HTuple* hv_ROI_Row2,
                             HTuple* hv_ROI_Col2)
{
    HObject ho_Gray, ho_RegionBlackRaw, ho_RegionBlackOpen;
    HObject ho_RegionBlackConnAll, ho_RegionBlackValid, ho_RegionBlackUnion;
    HObject ho_BandLeft, ho_BandRight, ho_BandTop, ho_BandBottom;
    HObject ho_LeftCand, ho_RightCand, ho_TopCand, ho_BottomCand;
    HObject ho_LeftConn, ho_RightConn, ho_TopConn, ho_BottomConn;
    HObject ho_LeftSel, ho_RightSel, ho_TopSel, ho_BottomSel;
    HObject ho_LeftMax, ho_RightMax, ho_TopMax, ho_BottomMax;
    HObject ho_ROI_Middle, ho_GrayROI_Domain;

    HTuple hv_BlackLow = 0, hv_BlackHigh = 16;
    HTuple hv_BandLeftRatio = 0.15, hv_BandRightRatio = 0.15;
    HTuple hv_BandTopRatio = 0.18, hv_BandBottomRatio = 0.18;
    HTuple hv_DefaultRow1Ratio = 0.22, hv_DefaultRow2Ratio = 0.78;
    HTuple hv_DefaultCol1Ratio = 0.10, hv_DefaultCol2Ratio = 0.90;
    HTuple hv_OpenSize = 5, hv_BorderMargin = 5, hv_MinBorderArea = 300;
    HTuple hv_MinROIHeight = 150, hv_MinROIWidth = 250;

    HTuple hv_Channels, hv_Width, hv_Height;
    HTuple hv_LeftNum, hv_RightNum, hv_TopNum, hv_BottomNum;
    HTuple hv_LR1, hv_LC1, hv_LR2, hv_LC2;
    HTuple hv_RR1, hv_RC1, hv_RR2, hv_RC2;
    HTuple hv_TR1, hv_TC1, hv_TR2, hv_TC2;
    HTuple hv_BR1, hv_BC1, hv_BR2, hv_BC2;

    CountChannels(ho_Image, &hv_Channels);
    if (0 != int(hv_Channels > 1)) {
        Rgb1ToGray(ho_Image, &ho_Gray);
    } else {
        ho_Gray = ho_Image;
    }

    GetImageSize(ho_Gray, &hv_Width, &hv_Height);

    Threshold(ho_Gray, &ho_RegionBlackRaw, hv_BlackLow, hv_BlackHigh);
    OpeningRectangle1(ho_RegionBlackRaw, &ho_RegionBlackOpen, hv_OpenSize, hv_OpenSize);
    Connection(ho_RegionBlackOpen, &ho_RegionBlackConnAll);
    SelectShape(ho_RegionBlackConnAll, &ho_RegionBlackValid, "area", "and",
                hv_MinBorderArea, 99999999);
    Union1(ho_RegionBlackValid, &ho_RegionBlackUnion);

    GenRectangle1(&ho_BandLeft, 0, 0, hv_Height - 1, (hv_Width * hv_BandLeftRatio).TupleRound());
    GenRectangle1(&ho_BandRight, 0, (hv_Width * (1.0 - hv_BandRightRatio)).TupleRound(),
                  hv_Height - 1, hv_Width - 1);
    GenRectangle1(&ho_BandTop, 0, 0, (hv_Height * hv_BandTopRatio).TupleRound(), hv_Width - 1);
    GenRectangle1(&ho_BandBottom, (hv_Height * (1.0 - hv_BandBottomRatio)).TupleRound(),
                  0, hv_Height - 1, hv_Width - 1);

    Intersection(ho_RegionBlackUnion, ho_BandLeft, &ho_LeftCand);
    Intersection(ho_RegionBlackUnion, ho_BandRight, &ho_RightCand);
    Intersection(ho_RegionBlackUnion, ho_BandTop, &ho_TopCand);
    Intersection(ho_RegionBlackUnion, ho_BandBottom, &ho_BottomCand);

    Connection(ho_LeftCand, &ho_LeftConn);
    Connection(ho_RightCand, &ho_RightConn);
    Connection(ho_TopCand, &ho_TopConn);
    Connection(ho_BottomCand, &ho_BottomConn);

    SelectShape(ho_LeftConn, &ho_LeftSel, "area", "and", hv_MinBorderArea, 99999999);
    SelectShape(ho_RightConn, &ho_RightSel, "area", "and", hv_MinBorderArea, 99999999);
    SelectShape(ho_TopConn, &ho_TopSel, "area", "and", hv_MinBorderArea, 99999999);
    SelectShape(ho_BottomConn, &ho_BottomSel, "area", "and", hv_MinBorderArea, 99999999);

    CountObj(ho_LeftSel, &hv_LeftNum);
    CountObj(ho_RightSel, &hv_RightNum);
    CountObj(ho_TopSel, &hv_TopNum);
    CountObj(ho_BottomSel, &hv_BottomNum);

    HTuple Row1 = (hv_Height * hv_DefaultRow1Ratio).TupleRound();
    HTuple Row2 = (hv_Height * hv_DefaultRow2Ratio).TupleRound();
    HTuple Col1 = (hv_Width * hv_DefaultCol1Ratio).TupleRound();
    HTuple Col2 = (hv_Width * hv_DefaultCol2Ratio).TupleRound();

    if (0 != int(hv_LeftNum > 0)) {
        SelectShapeStd(ho_LeftSel, &ho_LeftMax, "max_area", 70);
        SmallestRectangle1(ho_LeftMax, &hv_LR1, &hv_LC1, &hv_LR2, &hv_LC2);
        Col1 = hv_LC2 + hv_BorderMargin;
    }
    if (0 != int(hv_RightNum > 0)) {
        SelectShapeStd(ho_RightSel, &ho_RightMax, "max_area", 70);
        SmallestRectangle1(ho_RightMax, &hv_RR1, &hv_RC1, &hv_RR2, &hv_RC2);
        Col2 = hv_RC1 - hv_BorderMargin;
    }
    if (0 != int(hv_TopNum > 0)) {
        SelectShapeStd(ho_TopSel, &ho_TopMax, "max_area", 70);
        SmallestRectangle1(ho_TopMax, &hv_TR1, &hv_TC1, &hv_TR2, &hv_TC2);
        Row1 = hv_TR2 + hv_BorderMargin;
    }
    if (0 != int(hv_BottomNum > 0)) {
        SelectShapeStd(ho_BottomSel, &ho_BottomMax, "max_area", 70);
        SmallestRectangle1(ho_BottomMax, &hv_BR1, &hv_BC1, &hv_BR2, &hv_BC2);
        Row2 = hv_BR1 - hv_BorderMargin;
    }

    if (0 != int(Row1 < 0)) Row1 = 0;
    if (0 != int(Col1 < 0)) Col1 = 0;
    if (0 != int(Row2 > hv_Height - 1)) Row2 = hv_Height - 1;
    if (0 != int(Col2 > hv_Width - 1)) Col2 = hv_Width - 1;

    if (0 != HTuple(HTuple(HTuple(int(Row2 <= Row1).TupleOr(int(Col2 <= Col1)))
                  .TupleOr(int((Row2 - Row1) < hv_MinROIHeight)))
                  .TupleOr(int((Col2 - Col1) < hv_MinROIWidth)))) {
        Row1 = (hv_Height * hv_DefaultRow1Ratio).TupleRound();
        Row2 = (hv_Height * hv_DefaultRow2Ratio).TupleRound();
        Col1 = (hv_Width * hv_DefaultCol1Ratio).TupleRound();
        Col2 = (hv_Width * hv_DefaultCol2Ratio).TupleRound();
    }

    if (0 != int(Row1 < 0)) Row1 = 0;
    if (0 != int(Col1 < 0)) Col1 = 0;
    if (0 != int(Row2 > hv_Height - 1)) Row2 = hv_Height - 1;
    if (0 != int(Col2 > hv_Width - 1)) Col2 = hv_Width - 1;

    GenRectangle1(&ho_ROI_Middle, Row1, Col1, Row2, Col2);
    ReduceDomain(ho_Gray, ho_ROI_Middle, &ho_GrayROI_Domain);
    CropDomain(ho_GrayROI_Domain, ho_GrayROI);

    *hv_ROI_Row1 = Row1;
    *hv_ROI_Col1 = Col1;
    *hv_ROI_Row2 = Row2;
    *hv_ROI_Col2 = Col2;
}

void Preprocess_Gray_LR(HObject ho_Image,
                        HObject* ho_LeftROI,
                        HObject* ho_RightROI,
                        HTuple* l_r1,
                        HTuple* l_c1,
                        HTuple* l_r2,
                        HTuple* l_c2,
                        HTuple* r_r1,
                        HTuple* r_c1,
                        HTuple* r_r2,
                        HTuple* r_c2)
{
    HObject roi_full;
    HTuple R1, C1, R2, C2;
    Preprocess_Gray_WithROI(ho_Image, &roi_full, &R1, &C1, &R2, &C2);

    HTuple full_width = C2 - C1 + 1;
    HTuple mid = C1 + (full_width / 2.0).TupleFloor() - 1;
    if (0 != int(mid <= C1)) mid = C1 + 1;
    if (0 != int(mid >= C2)) mid = C2 - 1;

    *l_r1 = R1;
    *l_c1 = C1;
    *l_r2 = R2;
    *l_c2 = mid;
    *r_r1 = R1;
    *r_c1 = mid + 1;
    *r_r2 = R2;
    *r_c2 = C2;

    HObject gray;
    HTuple ch;
    CountChannels(ho_Image, &ch);
    if (0 != int(ch > 1)) {
        Rgb1ToGray(ho_Image, &gray);
    } else {
        gray = ho_Image;
    }

    HObject left_rect, right_rect, left_red, right_red;
    GenRectangle1(&left_rect, *l_r1, *l_c1, *l_r2, *l_c2);
    GenRectangle1(&right_rect, *r_r1, *r_c1, *r_r2, *r_c2);
    ReduceDomain(gray, left_rect, &left_red);
    ReduceDomain(gray, right_rect, &right_red);
    CropDomain(left_red, ho_LeftROI);
    CropDomain(right_red, ho_RightROI);
}
