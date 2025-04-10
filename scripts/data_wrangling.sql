SELECT * FROM metadata_cleaned_f;

SELECT Region, COUNT(*) as Count, Year
FROM metadata_cleaned_f
GROUP BY Region, Year
ORDER BY Year, Count DESC;

SELECT Stx, PT 
FROM metadata_cleaned_f;

-- Stx repetition check
SELECT Stx, COUNT(*) AS Stx_counter
FROM metadata_cleaned_f
WHERE Stx = "stx1a stx2a stx2c" OR Stx = "stx2a stx2c stx1a"
GROUP BY Stx;

-- Alright, now generate counts from updated dataset in Python

SELECT 
    Stx, 
    COUNT(Stx) AS Stx_Count, 
    PT, 
    COUNT(PT) AS PT_Count
FROM 
    fixed_pt_stx
GROUP BY 
    Stx, PT
ORDER BY 
    PT, Stx;
    
-- SELECT * FROM test.stx_pt_count;

SELECT Country, COUNT(*) AS Country_counter
FROM metadata_f
GROUP BY Country -- Portgual (2)
ORDER BY Country_counter ASC;

SELECT *
FROM metadata_f
WHERE Country = "Portgual";

SELECT COUNT(*)
FROM metadata_f
WHERE Country = "Portugal"; -- +2

SET SQL_SAFE_UPDATES = 0;
UPDATE metadata_f SET Country = 'Portugal' WHERE Country = 'Portgual';
UPDATE metadata_f SET Country = 'N' WHERE Country = 'Wales';
UPDATE metadata_f SET Country = 'UK' WHERE Country = 'N';
UPDATE metadata_f SET Stx = 'stx1a stx2a stx2c' WHERE Stx = 'stx2a stx2c stx1a';
UPDATE metadata_f SET Stx = 'stx2c stx1a' WHERE Stx = 'stx1a stx2c';
UPDATE metadata_f SET Stx = 'stx2a stx1a' WHERE Stx = 'stx1a stx2a';

SET SQL_SAFE_UPDATES = 1;

SELECT COUNT(*) FROM metadata_f
WHERE Country = "Portugal"; -- all fixed!

SELECT COUNT(*) FROM metadata_f
WHERE Country = "Wales"; -- good

/**SELECT COUNT(*) FROM metadata_f
WHERE Country = "N"; -- good**/ -- No Ns

SELECT COUNT(*) FROM metadata_f
WHERE Country = "UK";

SELECT COUNT(*) FROM metadata_f
WHERE Stx = "stx2a stx2c stx1a";

SELECT COUNT(*) FROM metadata_f
WHERE Stx = "stx1a stx2a stx2c"; -- NICE!

SELECT * FROM metadata_f;

SELECT COUNT(*) FROM metadata_f; -- 2874 cells -- needs cleaning again

SELECT *
FROM metadata_f
WHERE Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA'
ORDER BY Region;

-- Import new table and prepare data for individual heatmap (stx vs PT)
-- SELECT * FROM stx_pt_cluster;

SELECT 
    Stx2a, 
    COUNT(Stx2a) AS Stx2a_Count, 
    Stx2c, 
    COUNT(Stx2c) AS Stx2c_Count, 
    Stx2d, 
    COUNT(Stx2d) AS Stx2d_Count, 
    Stx1a, 
    COUNT(Stx1a) AS Stx1a_Count, 
    Stx1c, 
    COUNT(Stx1c) AS Stx1c_Count, 
    PT, 
    COUNT(PT) AS PT_Count
FROM 
    stx_pt_cluster
GROUP BY 
    Stx2a, Stx2c, Stx2d, Stx1a, Stx1c, PT
ORDER BY 
    PT, Stx2a, Stx2c, Stx2d, Stx1a, Stx1c;
    
SELECT Stx1a, Stx1c, Stx2a, Stx2c, Stx2d, PT
FROM stx_pt_cluster;

/**Check original metadata (XX50235metadata)
SELECT Region, COUNT(*) AS Reg_count
FROM XX50235metadata
WHERE Region = "Wales";**/

SELECT * 
FROM metadata_f
WHERE Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA';
  
SELECT Region, Count(*) AS Region_Count
FROM metadata_f
WHERE Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA'
GROUP BY Region
ORDER BY Region_Count DESC;

SELECT PT, Count(*) AS PT_Count
FROM metadata_f
WHERE Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA'
GROUP BY PT
ORDER BY PT_Count DESC;

SELECT Country, Count(*) AS Country_Count
FROM metadata_f
WHERE Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA'
GROUP BY Country
ORDER BY Country_Count DESC;

-- Temporal trends data

SELECT Region, Year, COUNT(*) AS Sample_Size
FROM metadata_f
WHERE Stx <> '-' 
  AND Stx <> 'NA' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
GROUP BY Region, Year
ORDER BY Year, Sample_Size DESC;

SELECT Country, Year, COUNT(*) AS Sample_Size
FROM metadata_f
WHERE Stx <> '-' 
  AND Stx <> 'NA' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
GROUP BY Country, Year
ORDER BY Year, Sample_Size DESC;

SELECT Stx, Count(*) AS Stx_Count
FROM metadata_f
WHERE Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA'
GROUP BY Stx
ORDER BY Stx_Count DESC;

-- EXTRACT NEW FINAL DATASET

SELECT *
FROM metadata_f
WHERE Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA';
  
  -- Extract Stx and PT counts
  
SELECT 
	Stx, 
    COUNT(Stx) AS Stx_Count, 
    PT, 
    COUNT(PT) AS PT_Count
FROM metadata_f
WHERE Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA'
GROUP BY Stx, PT
ORDER BY Stx_Count, PT_Count DESC;

-- separate UK samples from Non

SELECT 
    Stx,
    COUNT(*) AS Stx_Count,
    PT,
    COUNT(*) AS PT_Count
FROM metadata_f
WHERE Region = "UK"
  AND Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA'
GROUP BY Stx, PT;
  
SELECT 
    Stx,
    COUNT(*) AS Stx_Count,
    PT,
    COUNT(*) AS PT_Count
FROM metadata_f
WHERE Region != "UK"
  AND Stx <> '-' 
  AND PT NOT IN ('untypable', '#N/A', 'NA')
  AND Stx <> 'NA'
GROUP BY Stx, PT;
