def get_ontology_unifier(dataset_name, domains):
    return {'taskmaster': TaskmasterOntologyUnifier(domains),
            'schemaguided': SchemaGuidedOntologyUnifier(domains)
            }[dataset_name]


class OntologyUnifier():

    def __init__(self, domains, remove_underscores=True):
        super(OntologyUnifier, self).__init__()
        self._domains = domains
        self._domain_mapping = None
        self._slot_mapping = None
        self._remove_underscores = remove_underscores

    def _init_mappings(self):

        # substitute "_" with " " in domain and slot names
        if self._remove_underscores:

            new_domain_mapping = {}
            for k in self._domain_mapping.keys():
                new_domain_mapping[k.replace("_", " ")] = self._domain_mapping[k]
            self._domain_mapping = new_domain_mapping

            new_slot_mapping = {}
            for d, slots in self._slot_mapping.items():
                new_slot_mapping[d] = {}
                for k, v in slots.items():
                    new_slot_mapping[d][k] = v.replace("_", " ")
            self._slot_mapping = new_slot_mapping

        if self._domains is not None:
            # filter out unwanted target domains
            new_mapping = {}
            for k in self._domains:
                if k in self._domain_mapping.keys():
                    new_mapping[k] = self._domain_mapping[k]
            self._domain_mapping = new_mapping
        else:
            # get list of actual target domains
            self._domains = list(self._domain_mapping.keys())

        # prepare mappings
        self._original_domains = sum(self._domain_mapping.values(), [])
        self._original_new_domain_mapping = {x: key for key, value in self._domain_mapping.items() for x in value}

    def get_domains(self):
        return self._domains

    def get_original_domains(self):
        return self._original_domains

    def map_domain(self, original_domain):
        if original_domain in self._original_new_domain_mapping:
            return self._original_new_domain_mapping[original_domain]
        else:
            return original_domain

    def map_domain_reverse(self, mapped_domain):
        return self._domain_mapping[mapped_domain]

    def map_slot(self, original_slot, original_domain):
        if original_domain in self._slot_mapping and \
           original_slot in self._slot_mapping[original_domain]:
            return self._slot_mapping[original_domain][original_slot]
        else:
            if self._remove_underscores:
                slot = original_slot.replace("_", " ")
            if original_domain not in self._slot_mapping:
                self._slot_mapping[original_domain] = {}
            self._slot_mapping[original_domain][original_slot] = slot
            return slot

    def map_slot_reverse(self, mapped_slot, original_domain):
        pass


class SchemaGuidedOntologyUnifier(OntologyUnifier):

    def __init__(self, domains):
        super(SchemaGuidedOntologyUnifier, self).__init__(domains)

        # see schema_guided_dataloader.py for a complete list of domains
        self._domain_mapping = {
            'hotel':       ['Hotels_1', 'Hotels_2', 'Hotels_3', 'Hotels_4'],
            'train':       ['Trains_1'],
            'attraction':  ['Travel_1'],
            'restaurant':  ['Restaurants_1', 'Restaurants_2'],
            'taxi':        ['RideSharing_1', 'RideSharing_2'],
            'bus':         ['Buses_1', 'Buses_2', 'Buses_3'],
            'flight':      ['Flights_1', 'Flights_2', 'Flights_3', 'Flights_4'],
            'music':       ['Music_1', 'Music_2', 'Music_3'],
            'movie':       ['Media_1', 'Media_2', 'Media_3', 'Movies_1', 'Movies_2', 'Movies_3'],
            'service':     ['Services_1', 'Services_2', 'Services_3', 'Services_4'],
            'bank':        ['Banks_1', 'Banks_2', 'Payment_1'],
            'event':       ['Events_1', 'Events_2', 'Events_3'],
            'rentalcar':   ['RentalCars_1', 'RentalCars_2', 'RentalCars_3'],
            'apartment':   ['Homes_1', 'Homes_2'],
            'calendar':    ['Calendar_1'],
            'weather':     ['Weather_1'],
            'alarm':       ['Alarm_1'],
            'messaging':   ['Messaging_1']
        }

        self._slot_mapping = {
            'Hotels_1': {
                'star_rating':    'stars',
                'hotel_name':      'name',
                'number_of_days':   'stay',
                'has_wifi':        'internet',
                'check_in_date':   'date',
                'street_address': 'address',
                'phone_number': 'phone',
                'price_per_night': 'price_range',
                'stay_length': 'stay',
            },
            'Hotels_2': {
                'hotel_name': 'name',
                'where_to':   'destination',
                'check_in_date':   'check_in',
                'check_out_date':   'check_out',
                'street_address': 'address',
                'phone_number': 'phone',
                'price_per_night': 'price_range',
                'stay_length': 'stay',
            },
            'Hotels_3': {
                'hotel_name': 'name',
                'check_in_date':   'check_in',
                'check_out_date':   'check_out',
                'street_address': 'address',
                'phone_number': 'phone',
                'stay_length': 'stay',
            },
            'Hotels_4': {
                'star_rating':    'stars',
                'place_name':      'name',
                'check_in_date':   'check_in',
                'street_address': 'address',
                'phone_number': 'phone',
                'price_per_night': 'price_range',
                'stay_length': 'stay',
            },
            'Trains_1': {
                'journey_start_time':  'leave_at',
                'to':                  'destination',
                'from':                 'departure'
            },
            'Travel_1':      {'category': 'type',
                              'attraction_name': 'name',
                              'phone_number': 'phone',
                              },
            'RideSharing_1': {'number_of_riders': 'people'},
            'Buses_1': {
                'from_location':  'departure',
                'to_location':    'destination',
                'leaving_time':    'leave_at',
                'leaving_date':   'date',
                'travelers':       'people',
                'phone_number': 'phone',
            },
            'Buses_2': {
                'origin':          'departure',
                'departure_time':   'leave_at',
                'group_size':       'people',
                'departure_date':   'date',
                'phone_number': 'phone',
            },
            'Buses_3': {
                'from_city':  'departure',
                'to_city':    'destination',
                'departure_time':    'leave_at',
                'departure_date':   'date',
                'phone_number': 'phone',
            },
            'Restaurants_1': {
                'price_range':     'price_range',
                'restaurant_name': 'name',
                'cuisine':          'food',
                'phone_number': 'phone',
            },
            'Restaurants_2': {
                'price_range':     'price_range',
                'restaurant_name': 'name',
                'category':         'food',
                'phone_number': 'phone',
            },
            'Music_3': {
                'track':     'song_name',
                'device':    'playback_device'
            },
            'Movies_3': {
                'directed_by':     'director',
                'cast':            'starring'
            },
            'Media_1': {
                'title':       'movie_name',
                'directed_by':     'director',
                'device':    'playback_device'
            },

            'Flights_1': {
                'outbound_departure_time': 'leave at',
                'inbound_departure_time': 'leave at',
            },
            'Flights_2': {
                'outbound_departure_time': 'leave at',
                'inbound_departure_time': 'leave at',
            },
            'Flights_3': {'number_checked_bags': 'people',
                          'phone_number': 'phone',
                          'outbound_departure_time': 'leave at',
                          'inbound_departure_time': 'leave at',
                          },
            'Media_2': {'actors': 'starring'},
            'Media_3': {'title': 'name'},
            'Banks_1': {'recipient_account_name': 'recipient'},
            'Banks_2': {'transfer_amount': 'amount'},
            'Payment_1': {'receiver': 'receiver'},
            'Events_1': {
                'city_of_event': 'city',
                'event_name': 'name',
                'number_of_seats': 'people'
            },
            'Events_2': {
                'event_name': 'name',
                'event_type': 'type',
            },
            'Events_3': {
                'event_name': 'name',
                'event_type': 'type',
            },
            'RentalCars_1': {
                'car_type': 'car',
                'pickup_date': 'date',
                'pickup_location': 'departure',
                'total_price': 'fee',
                'pickup_time': 'time',
                'dropoff_time': 'time',
                'dropoff_location': 'destination',
                'dropoff_date': 'date',
                'car_name': 'car',
            },
            'RentalCars_2': {
                'car_type': 'car',
                'car_type': 'car',
                'pickup_date': 'date',
                'pickup_location': 'departure',
                'total_price': 'fee',
                'pickup_time': 'time',
                'dropoff_time': 'time',
                'dropoff_location': 'destination',
                'dropoff_date': 'date',
                'car_name': 'car',
            },
            'RentalCars_3': {
                'start_date':  'pickup_date',
                'end_date':    'dropoff_date',
                'city':        'pickup_city',
                'car_type':    'car',
                'pickup_date': 'date',
                'pickup_location': 'departure',
                'total_price': 'fee',
                'pickup_time': 'time',
                'dropoff_time': 'time',
                'dropoff_location': 'destination',
                'dropoff_date': 'date',
                'car_name': 'car',
            },
        }

        self._init_mappings()


class TaskmasterOntologyUnifier(OntologyUnifier):

    def __init__(self, domains):
        super(TaskmasterOntologyUnifier, self).__init__(domains)

        # see taskmaster_dataloader.py for a complete list of domains
        self._domain_mapping = {
            'hotel': ['hotel_search', 'hotel1_detail', 'hotel2_detail',
                      'hotel4_detail', 'hotel3_detail', 'hotel_booked'],
            'restaurant': ['restaurant', 'restaurant_reservation'],
            'flight': ['flight_search', 'flight1_detail', 'flight2_detail',
                       'flight3_detail', 'flight4_detail', 'flight_booked'],
            'sport': ['epl', 'nba', 'mlb', 'nfl', 'mls'],
            'music': ['music'],
            'movie': ['movie_search', 'movie_ticket'],
            'food': ['food_order']
        }

        self._slot_mapping = {
            'hotel_search': {
                'price_range': 'price_range',
                'star_rating': 'stars',
                'sub_location': 'sub-location',
                'num': 'people'
            },
            'hotel2_detail': {
                'type':  'type',
                'star_rating': 'stars'
            },
            'hotel4_detail': {
                'type':  'type',
                'star_rating': 'stars'
            },
            'hotel3_detail': {
                'type':  'type',
                'star_rating': 'stars'
            },
            'hotel1_detail': {
                'type':  'type',
                'star_rating': 'stars'
            },
            'restaurant': {
                'price_range': 'price_range',
                'location': 'city',
                'type': 'food',
                'rating': 'stars',
                'num': 'people',
                'sub-location': 'area',
            },
            'music': {
                'name': 'artist',
                'type': 'genre',
                'describes_type': 'type_description',
                'describes_album': 'album_description',
                'describes_track': 'track_description',
                'describes_genre': 'genre_description',
                'describes_artist': 'artist_description',
                'technical_difficulty': 'difficulty_description',
            },
            'movie_search': {
                'name': 'name',
                'show_time': 'time'
            },
            'movie_ticket': {
                'name': 'name',
                'show_time': 'time',
                'num': 'people'
            },
            'food_order': {
                'rating': 'stars'
            },
            'flight_search': {
                'destination1': 'destination',
                'destination2': 'destination',
                'num': 'people',
                'origin': 'departure',
                'from': 'leave at',
                'to': 'arrive by',
                'stops': 'destination',
                'time_of_day': 'time',
                'fare': 'fee',
                'total_fare': 'fee',

            }
        }

        self._init_mappings()
