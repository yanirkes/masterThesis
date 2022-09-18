import matplotlib.colors as mcolors

class constant():

    missing_ids_webscrapping = [71, 6, 21, 7, 46, 1, 47, 32, 68, 11]
    team_team_dict = {'Baloncesto Fuenlabrada SAD': ['Alta Gestión Fuenlabrada', 'ALTA GESTIÓN Fuenlabrada',
                                                     'Ayuda en Acción Fuenlabrada', 'Baloncesto Fuenlabrada',
                                                     'Jabones Pardo Fuenlabrada', 'Mad-Croc Fuenlabrada'],
                      'Baloncesto Málaga SAD': ['Unicaja'],
                      'Basket Zaragoza 2002 SAD': ['CAI Zaragoza'],
                      'Bàsquet Manresa SAD': ['Assignia Manresa', 'Bàsquet Manresa', 'Suzuki Manresa', 'Ricoh Manresa'],
                      'C Estudiantes SAD': ['Adecco Estudiantes', 'Asefa Estudiantes', 'ASEFA Estudiantes',
                                            'MMT Estudiantes'],
                      'C Joventut Badalona SAD': ['DKV Joventut', 'FIATC Joventut', 'FIATC Mutua Joventut'],
                      'C.B. Breogan SAD': ['Leche Rio', 'Leche Río'],
                      'C.B. Girona SAD': ['Akasvayu Girona', 'Casademont Girona'],
                      'C.B. Leon SAD': ['Grupo Begar León'],
                      'C.D. Basket Bilbao Berri SAD': ['Bizkaia Bilbao Basket', 'iurbentia Bilbao Basket',
                                                       'Lagun Aro Bilbao Basket', 'Uxue Bilbao Basket',
                                                       'Gescrap Bizkaia'],
                      'C.E. Lleida Basquetbol SAD': ['Caprabo Lleida', 'Plus Pujol Lleida'],
                      'CB 1939 Canarias SAD': ['CB Canarias'],
                      'CB Gran Canaria-Claret SAD': ['Gran Canaria 2014', 'Gran Canaria Grupo Dunas', 'Gran Canaria',
                                                     'Herbalife Gran Canaria', 'Kalise Gran Canaria','Auna G. Canaria'],
                      'CB Granada SAD': ['CB Granada'],
                      'CB Valladolid SAD': ['Blancos de Rueda Valladolid', 'Forum Valladolid', 'Fórum Valladolid',
                                            'Grupo Capitol Valladolid'],
                      'Donosti Gipuzkoa Basket 2001 SKE SAD': ['Bruesa GBC', 'Lagun Aro GBC'],
                      'Futbol Club Barcelona': ['AXA FC Barcelona', 'FC Barcelona Regal', 'FC Barcelona',
                                                'Regal FC Barcelona', 'Winterthur FC Barcelona'],
                      'Lucentum B Alicante SAD': ['Etosa Alicante', 'Lucentum Alicante', 'Meridiano Alicante'],
                      'Menorca  SAD': ['Llanera Menorca', 'Menorca Basquet', 'ViveMenorca'],
                      'Obradoiro CAB SAD': ['Blusens Monbus', 'Xacobeo Blu:Sens'],
                      'Real Betis Baloncesto SAD': ['Cajasol', 'Banca Civica', 'Caja San Fernando'],
                      'Real Madrid C de F': ['Real Madrid'],
                      'Saski Baskonia SAD': ['Caja Laboral', 'TAU Cerámica'],
                      'Tenerife Baloncesto SAD': ['Unelco Tenerife'],
                      'UCAM Murcia Club Baloncesto SAD': ['CB Murcia', 'Club Baloncesto Murcia',
                                                          'Polaris World CB Murcia', 'Polaris World Murcia',
                                                          'UCAM Murcia CB', 'UCAM Murcia'],
                      'Valencia Basket Club SAD': ['Pamesa Cerámica Valencia', 'Pamesa Valencia',
                                                   'Power Electronics Valencia', 'Valencia Basket Club',
                                                   'Valencia Basket']}

    team_stadium = {'Baloncesto Fuenlabrada SAD': ["Pabellón Fernando Martín"],
    'Baloncesto Málaga SAD': ["Palacio de Deportes José María Martín Carpena"],
    'Basket Zaragoza 2002 SAD': ["Pabellón Príncipe Felipe"],
    'Bàsquet Manresa SAD': ["Pabellón Nou Congost"],
    'C Estudiantes SAD': ["WiZink Center"],
    'C Joventut Badalona SAD': ["Palau Olímpic"],
    'C.B. Breogan SAD': ["Pazo dos Deportes"],
    'C.B. Girona SAD': ["Palau Girona-Fontajau"],
    'C.B. Leon SAD': ["Municipal de los Deportes"],
    'C.D. Basket Bilbao Berri SAD': ["Bilbao Arena"],
    'C.E. Lleida Basquetbol SAD': ["Pavelló Barris Nord"],
    'CB 1939 Canarias SAD': ["Pabellón Santiago Martín San Cristóbal de La Laguna"],
    'CB Gran Canaria-Claret SAD': ["Estadio de Gran Canaria"],
    'CB Granada SAD': ["Palacio Municipal de los Deportes de Granada"],
    'CB Valladolid SAD': ["Polideportivo Pisuerga"],
    'Donosti Gipuzkoa Basket 2001 SKE SAD': ["Donostia Arena 2016"],
    'Futbol Club Barcelona': ["Palau Blaugrana"],
    'Lucentum B Alicante SAD': ["Pabellón Pedro Ferrándiz"],
    'Menorca  SAD': ["Pavelló Menorca"],
    'Obradoiro CAB SAD': ["Multiusos Fontes do Sar"],
    'Real Betis Baloncesto SAD': ["Palacio de Deportes San Pablo"],
    'Real Madrid C de F': ["WiZink Center"],
    'Saski Baskonia SAD': ["Saski Baskonia SAD"],
    'Tenerife Baloncesto SAD': ["Pabellón Santiago Martín San Cristóbal de La Laguna"],
    'UCAM Murcia Club Baloncesto SAD': ["Palacio de Deportes de Murcia"],
    'Valencia Basket Club SAD': ["Pabellón Fuente de San Luis"],
    }

    colors_name = list(mcolors.CSS4_COLORS.keys())
    colors_number = list(mcolors.CSS4_COLORS.values())

    query_ml_actions_value = q_for_tbl = """
    select year
          , game_id
          , team
          , home_away
          , time_marker
          , five_on_court
          , score_dif
          , shot_score
          , shot_miss
          , drb
          , orb
          , stl
          , turn
          , foul_made
          , foul_gain
          , quarter
          , action_status_dir
          , action_status_sum
          , action_status_canasta
          , is_good__dir_actions 
          , is_good_sum_actions 
    from basket.data_analytics_all_metrics_agg
    where year between 2003 and 2005
    and score_dif is not null
    and time_marker in (select distinct time_marker from basket.score_by_mintues_temp) 
    """

    query_ml_oloc = """select a_.year,
                    a_.game_id,
                    a_.time_marker,
                     case 
                        when a_.quarter = 4 then 'q4' 
                        when a_.quarter = 3 then 'q3' 
                        when a_.quarter = 2 then 'q2' 
                        else 'q1'
                        end AS quarter,
                    a_.q_minutes,
                    a_.team,
                    case
                        when a_.five_on_court = 0 then 1 
                        else a_.five_on_court
                    end as oloc,
                    a_.score_dif,
                    a_.shot_score,
                    a_.shot_miss,
                    a_.foul_made,
                    a_.foul_gain,
                    a_.home_away,
                    case when sub_after_shot_by_team = 1 then 'sub_after_shot_1' else 'sub_after_shot_0' end as sub_after_shot_by_team ,
                    case when sub_after_miss_by_team = 1 then 'sub_after_miss_1' else 'sub_after_miss_0' end as sub_after_miss_by_team ,
                    case when sub_after_foul = 1 then 'sub_after_foul_1' else 'sub_after_foul_0' end as sub_after_foul ,
                    cv_score_dif,
                    abs(cv_score_dif) as abs_cv_score_dif,
                    case
                      when (abs(cv_score_dif) >= 5) then 'Very high'
                      when (abs(cv_score_dif) < 5 and abs(cv_score_dif) >= 2.5) then 'High'
                      when (abs(cv_score_dif) < 2.5 and abs(cv_score_dif) >= 1.5) then 'Mid'
                      when (abs(cv_score_dif) < 1.5 and abs(cv_score_dif) >= 0.5) then 'Low'
                      else 'Very low'
                    end as group_cv
                    , case 
                        when cluster = 3 then 'high_team' 
                        when cluster = 2 then 'med_team' 
                        else 'low_team'
                        end as cluster
                    from basket.data_analytics_all_metrics_agg a_
                    left join 
                    (select distinct year, game_id, time_marker, team,
                    case 
                        when (action in ('Entra a Pista','Sale a Banquillo') and action_p_1_by_team in ('Mate','Canasta de 3','Canasta de 2','Canasta de 1')) then 1
                        else 0 	
                    end sub_after_shot_by_team,
                    case 
                        when (action in ('Entra a Pista','Sale a Banquillo') and action_p_1_by_team in ('Intento fallado de 1', 'Intento fallado de 2','Intento fallado de 3')) then 1
                        else 0 	
                    end sub_after_miss_by_team,
                    case 
                        when (action in ('Entra a Pista','Sale a Banquillo') and action_p_1_by_team in ('Falta Personal','Falta recibida')) then 1
                        else 0 	
                    end sub_after_foul
                    from(
                        select year, game_id, play_number, team , time_marker, action
                        , lag(action,-1) over(partition by year, game_id, team order by play_number asc) action_p_1_by_team
                        , lag(action,-1) over(partition by year, game_id order by play_number asc) action_p_1
                        from stg.all_data_info
                        -- where year = 2003
                        where team !='NA'
                        order by 1,2,3 asc
                    ) as d_
                    order by 1,2,3) as b_
                    on a_.year = b_.year
                    and a_.game_id = b_.game_id
                    and a_.time_marker = b_.time_marker
                    
                    left join 
                    basket.score_dif_cv_per_game c_
                    on a_.year = c_.year
                    and a_.game_id = c_.game_id
                    
                    left join 
                    basket.team_info d_
                    on a_.team = d_.team
                    
                    where score_dif is not null
                    and a_.year <= 2003
                    --limit 10000
                    """
