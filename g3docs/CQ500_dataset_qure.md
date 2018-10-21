## Intro:
- [qure.ai](http://headctstudy.qure.ai/) has collected a dataset of 313,318 CT scans. Out of which 21,095 scans were used to validate and the rest to develop the algorithm.

- Additionally, they also collected **CQ500 dataset** to 'clinically validate' the algorithm. This contains **491 scans with 193,317 slices**.
Only CQ500 dataset is publicly available.

- Their algorithm can anatomically locate (using  brain anatomy segmentation algorithms they had developed for BRATS at MICCIA 16 where they placed third. For more [info](http://blog.qure.ai/notes/brain-anatomy-segmentation)) and quantify hemorrhages. 
  
- With CQ500 dataset, we can only quantify hemorrhages. This is essentially a **multi-label classification** problem. 
## Annotation format used in CQ500:

The set of target labels (and corresponding acronyms as mentioned in [reads.xlsx](/media/reads.xlsx) file) are:

- Hemorrhage
  -   Intracranial
hemorrhage (ICH)
  - Intraparenchymal (IPH)
  - Intraventricular (IVH)
  - Subdural (SDH)
  - Extradural (EDH)
  - Subarachnoid (SAH)
- Fracture
  - Calvarial fracture
  - Other fracture
- Mass Effect
- Midline Shift
- BleedLocation-Left
- BleedLocation-Right
- Chronic Bleed

The gold standard for CQ500 dataset was **consensus of **three** independent radiologists**. So each of the above target labels has 3 reads e.g. R1: ICH, R2: ICH and R3: ICH.

The expected output from the model trained on CQ500 is the prediction probabilities of the above target labels as seen in [prediction_probabilites.xlsx](/media/prediction_probabilites.xlsx)

For more [info](/media/supplement.pdf) on CQ500 dataset.

---
### Sidenote
- The CQ500 dataset is more likely to be accurate as the reads were done by three radiologists with an experience of 8, 12 and 20 years in cranial CT interpretation respectively. This is essential as mentioned in a blog on medium,
> What is ground truth? Radiologists often disagree significantly on the segmentation or diagnosis called for by an MRI. Deep learning models can often deal with random variability in ground truth labels, but any systemic bias in radiology will persist in deep learning models trained on radiologists’ predictions.
  
- The CQ500 dataset is most likely the largest brain CT scan dataset publicly available. I think we should leverage this dataset by sticking to a similar format and possibly consider fewer target labels (compared to CQ500) based on the needs/ease of annotation for the doctors in JIPMER. 

- [MicroDicom viewer](http://www.microdicom.com/downloads.html)
was used for viewing DICOM files in CQ500 dataset (26.6 GB) and also for fast conversion to jpeg format.




